# -*- coding: utf-8 -*-
"""
Parses an XML file in order to evaluate sites
"""

__author__ = "Noah W. Garfinkle"
__copyright__ = "Copyright 2018, Noah W. Garfinkle"
__credits__ = ["Dr. Ximing Cai", "Dr. George Calfas", "Thomas 'Max' Foltz",
                    "Juliana McMillan-Wilhoit", "Matthew Hiett",
                    "Dylan Pasley", "Marcus Voegle", "Eric Kreiger"]
__license__ = "GPL"
__version__ = "0.0.1"
__version_dinosaur__ = "Apotosauras"
__maintainer__ = "Noah Garfinkle"
__email__ = "garfink2@illinois.edu"
__status__ = "Development"
__python_version__ = "2.7"
__date_created__ = "10 FEBRUARY 2018"

## IMPORTS
from lxml import etree as ET
import gdaltools as gdt
from osgeo import gdal, osr, ogr
import pandas as pd
import geopandas as gpd
import fiona
import pyproj
import rasterio
import numpy as np
import numpy.ma as ma
import math
import shapely
from shapely.geometry import Point, Polygon
from shapely.wkt import loads
import shapely.geometry as geom
import shapely.affinity
from rasterstats import zonal_stats, raster_stats, point_query, utils
import matplotlib.pyplot as plt
import datetime
import time

import CandidateDataFrameOperations as candidates
reload(candidates)
import ENSITEIO as eio
reload(eio)
import Objective_Analytic as objective_analytic
reload(objective_analytic)
import Objective_Raster as objective_raster
reload(objective_raster)
import Objective_Vector as objective_vector
reload(objective_vector)
import pgdissroute as pgdissroute
reload(pgdissroute)
import SpatialIO as io
reload(io)
import SpatialOpt as opt
reload(opt)

## HELPFUL FOR DEBUGGING
# %matplotlib inline
# pd.options.display.max_columns = 300

## SETUP
validStats = utils.VALID_STATS

""" REFERENCES
http://lxml.de/tutorial.html
https://mapbox.s3.amazonaws.com/playground/perrygeo/rasterio-docs/cookbook.html#rasterizing-geojson-features
http://skipperkongen.dk/2012/03/06/hello-world-of-raster-creation-with-gdal-and-python/
https://github.com/mapplus/qgis-scripts/blob/master/scripts/Raster%20Euclidean%20Distance%20Analysis.py
https://stackoverflow.com/questions/30740046/calculate-distance-to-nearest-feature-with-geopandas
http://portolan.leaffan.net/creating-sample-points-with-ogr-and-shapely-pt-2-regular-grid-sampling/
https://gis.stackexchange.com/questions/159624/converting-large-rasters-to-numpy
https://gis.stackexchange.com/questions/264793/crop-raster-in-memory-with-python-gdal-bindings
http://pythonhosted.org/rasterstats/manual.html#zonal-statistics -> can define custom statistics!!!
https://github.com/perrygeo/python-rasterstats/blob/master/src/rasterstats/main.py
https://github.com/perrygeo/python-rasterstats/blob/master/src/rasterstats/utils.py
https://github.com/SALib/SALib -> Sensitivity analysis software
"""

## FUNCTIONS
def returnCriteriaMetadataForMCDA(criteriaRow):
    criteriaName = criteriaRow.attrib["criteriaName"]
    scores = criteriaRow.find("Scores")
    weight = scores.attrib["weight"]
    isZeroExclusionary = scores.attrib["isZeroExclusionary"]
    return criteriaName,weight,isZeroExclusionary

def evaluateXML(xmlPath,returnDFInsteadOfLayerID=True,limitReturn=True):
    """ Runs site search for a given xml document

        Evaluation function

    Args:
        xmlPath (str): Path to input.xml

    Returns
        layerID (list<int>): The returned layer id for each evaluation in the
            database

    Raises:
        None

    Todo:
        * Currently rewritten to only write polygon layers

    Tests:
        > layerIDs = runMSSPIX("C:/Users/RDCERNWG/Documents/GIT/FLW_Missouri Mission Folder/RESULTS/Airfield 7.xml")
    """
    tree = ET.parse(xmlPath)
    root = tree.getroot()

    resultDir = root.attrib['resultDir']
    studyID = root.attrib['studyID']
    epsg = root.attrib['epsg']
    print "%s %s %s" %(resultDir,studyID,epsg)

    print "Section: Site Searches"
    siteSearches = root.find("SiteSearches")
    layerIDs = []
    evaluationDFs = []
    searchID = 1
    for siteSearch in siteSearches:
        siteSearch_studyObjectiveID = siteSearch.attrib['studyObjectiveID']
        siteSearch_layerID = siteSearch.attrib['layerID']
        siteSearch_type = siteSearch.attrib['type']
        siteSearch_name = siteSearch.attrib['name']
        siteSearch_note = siteSearch.attrib['note']
        siteSearch_nReturn = siteSearch.attrib['nReturn']

        print "Beginning site search %s of %s: %s" %(searchID,len(siteSearches),siteSearch_name)
        siteConfiguration = siteSearch.find("SiteConfiguration")[0]
        if siteConfiguration.tag == "WKTTemplate":
            print "WKT Template"

        searchParameters = siteSearch.find("SearchParameters")[0]
        if searchParameters.tag == "GriddedSearch":
            evaluationDF = candidates.buildGriddedSearchFromXML(siteConfiguration,searchParameters)

        if searchParameters.tag == "SingleSiteSearch":
            evaluationDF = candidates.buildSingleSiteSearchFromXML(siteConfiguration,searchParameters)

        if len(evaluationDF.index) == 0:
            print "Area and search parameters did not generate any candidates"
            return "No Results Found"


        siteEvaluation = siteSearch.find("SiteEvaluation")
        weights = []
        qafNames = []
        scoringDict = {}
        criteriaCount = 0
        for criteriaRow in siteEvaluation:
            criteriaCount += 1
            if len(evaluationDF.index) == 0:
                print "No Results Found"
                return "No Results Found"
            # set the column name if none
            if criteriaRow.attrib["criteriaName"] == "None":
                criteriaRow.attrib["criteriaName"] = "Criteria_%s" %(criteriaCount)

            # Get the metadata needed for scoring
            criteriaName,weight,isZeroExclusionary = returnCriteriaMetadataForMCDA(criteriaRow)
            qafName = "%s_QAF" %(criteriaName)
            weights.append(float(weight))
            qafNames.append(qafName)
            scoringDict[qafName] = float(weight)

            # Parse based on type of criteria
            try:
                if criteriaRow.tag == "CategoricalRasterStat":
                    startingSize = len(evaluationDF.index)
                    startingTime = datetime.datetime.now()
                    evaluationDF = objective_raster.buildCategoricalRasterStatFromXML(evaluationDF,criteriaRow)
                    endingTime = datetime.datetime.now()
                    endingSize = len(evaluationDF.index)
                    timeElapsed = endingTime - startingTime
                    print "Categorical Raster Stat: %s. Processed %s candidates in %s seconds, retaining %s candidates" %(criteriaRow.attrib['criteriaName'], startingSize, timeElapsed.seconds, endingSize)
                if criteriaRow.tag == "ContinuousRasterStat":
                    startingSize = len(evaluationDF.index)
                    startingTime = datetime.datetime.now()
                    evaluationDF = objective_raster.buildContinuousRasterStatFromXML(evaluationDF,criteriaRow)
                    endingTime = datetime.datetime.now()
                    endingSize = len(evaluationDF.index)
                    timeElapsed = endingTime - startingTime
                    print "Continuous Raster Stat: %s. Processed %s candidates in %s seconds, retaining %s candidates" %(criteriaRow.attrib['criteriaName'], startingSize, timeElapsed.seconds, endingSize)
                if criteriaRow.tag == "DistanceFromVectorLayer":
                    startingSize = len(evaluationDF.index)
                    startingTime = datetime.datetime.now()
                    evaluationDF = objective_vector.buildDistanceFromVectorLayerFromXML(evaluationDF,criteriaRow)
                    endingTime = datetime.datetime.now()
                    endingSize = len(evaluationDF.index)
                    timeElapsed = endingTime - startingTime
                    print "Vector Layer: %s. Processed %s candidates in %s seconds, retaining %s candidates" %(criteriaRow.attrib['criteriaName'], startingSize, timeElapsed.seconds, endingSize)
                if criteriaRow.tag == "CutFill":
                    startingSize = len(evaluationDF.index)
                    startingTime = datetime.datetime.now()
                    evaluationDF = objective_analytic.buildCutFillFromXML(evaluationDF,criteriaRow)
                    endingTime = datetime.datetime.now()
                    endingSize = len(evaluationDF.index)
                    timeElapsed = endingTime - startingTime
                    print "Cut Fill: %s. Processed %s candidates in %s seconds, retaining %s candidates" %(criteriaRow.attrib['criteriaName'], startingSize, timeElapsed.seconds, endingSize)
            except Exception as e:
                print "exception hit on criteria row"
                print e
                return criteriaRow

        # build the weights
        totalWeight = sum(weights)
        weightedMCDAColumns = []
        for qafName in qafNames:
            scoringDict[qafName] /= totalWeight
            assignedWeight = scoringDict[qafName]
            weightedMCDAColumn = "%s_weighted" %(qafName)
            weightedMCDAColumns.append(weightedMCDAColumn)
            evaluationDF[weightedMCDAColumn] = evaluationDF[qafName] * assignedWeight

        # Build the total score
        evaluationDF["MCDA_SCORE"] = 0
        for weightedMCDAColumn in weightedMCDAColumns:
            evaluationDF["MCDA_SCORE"] += evaluationDF[weightedMCDAColumn]

        # Build the standardized score
        maxScore = max(evaluationDF["MCDA_SCORE"])
        evaluationDF["MCDA_SCORE_STANDARDIZED"] = evaluationDF["MCDA_SCORE"] / maxScore * 100.0

        # implement nreturn
        if limitReturn:
            evaluationDF = evaluationDF.sort_values(by="MCDA_SCORE",ascending=False).head(int(siteSearch_nReturn))
            evaluationDF.reset_index()
            print "Completed site search %s of %s for %s.  Returned top %s candidates of %s." %(searchID,len(siteSearches),siteSearch_name,int(siteSearch_nReturn),endingSize)
        evaluationDFs.append(evaluationDF)

        ensiteLayerName = "%s_%s" %(siteSearch_name,time.strftime("%Y_%m_%d_%H_%M_%S"))
        if not returnDFInsteadOfLayerID:
            layerID = io.dataFrameToENSITEDatabase(evaluationDF,studyID,ensiteLayerName)
            layerIDs.append(layerID)
        searchID += 1

    # SITE RELATIONAL CONSTRAINTS
    print "Section: Site Relational Constraints"
    siteRelationalConstraints = root.find("SiteRelationalConstraints")
    individual = [22,4]
    scoreDF = opt.evaluate(individual,evaluationDFs,siteRelationalConstraints)
    evaluationDFs.append(scoreDF)
    """
    for siteRelationalConstraint in siteRelationalConstraints:
        if siteRelationalConstraint.tag == "SiteRelationalConstraint_Routing":
            print "Routing distance test"
            siteRelationalConstraint_constraintName = siteRelationalConstraint.attrib['constraintName']
            siteRelationalConstraint_candidate1TableIndex = int(siteRelationalConstraint.attrib['candidate1TableIndex'])
            siteRelationalConstraint_candidate1Index = int(siteRelationalConstraint.attrib['candidate1Index'])
            siteRelationalConstraint_candidate2TableIndex = int(siteRelationalConstraint.attrib['candidate2TableIndex'])
            siteRelationalConstraint_candidate2Index = int(siteRelationalConstraint.attrib['candidate2Index'])
            siteRelationalConstraint_note = siteRelationalConstraint.attrib['note']
            routingDistance = opt.evaluateCandidates_DrivingDistance(evaluationDFs[siteRelationalConstraint_candidate1TableIndex],siteRelationalConstraint_candidate1Index,evaluationDFs[siteRelationalConstraint_candidate2TableIndex],siteRelationalConstraint_candidate2Index)
            print routingDistance
        if siteRelationalConstraint.tag == "SiteRelationalConstraint_Euclidean":
            print "Euclidean distance test"
            siteRelationalConstraint_constraintName = siteRelationalConstraint.attrib['constraintName']
            siteRelationalConstraint_candidate1TableIndex = int(siteRelationalConstraint.attrib['candidate1TableIndex'])
            siteRelationalConstraint_candidate1Index = int(siteRelationalConstraint.attrib['candidate1Index'])
            siteRelationalConstraint_candidate2TableIndex = int(siteRelationalConstraint.attrib['candidate2TableIndex'])
            siteRelationalConstraint_candidate2Index = int(siteRelationalConstraint.attrib['candidate2Index'])
            siteRelationalConstraint_note = siteRelationalConstraint.attrib['note']
            euclideanDistance = opt.evaluateCandidates_EuclideanDistance(evaluationDFs[siteRelationalConstraint_candidate1TableIndex],siteRelationalConstraint_candidate1Index,evaluationDFs[siteRelationalConstraint_candidate2TableIndex],siteRelationalConstraint_candidate2Index)
            print euclideanDistance
    """
    if returnDFInsteadOfLayerID:
        return evaluationDFs
    else:
        return layerIDs
