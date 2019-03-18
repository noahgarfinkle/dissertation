# -*- coding: utf-8 -*-
"""
Manages simple objective functions for vector data layers
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
__date_created__ = "10 May 2018"

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
import ENSITEIO as eio
import Objective_Analytic as objective_analytic
import Objective_Raster as objective_raster
# import pgdissroute as pgdissroute
import SpatialIO as io
import SpatialOpt as opt

## VECTOR OPERATIONS
def filterByVectorBufferDistance(dfToFilter,vectorFilePath,bufferDistance,removeIntersected=True):
    """ Utilizes shapely hack to include/exclude buffers faster than Euclidean distance

    This produces the fundamental solution data structure of First Pass, a
    GeoPandas GeoDataFrame sized and rotated as specified, and falling exclusively
    within the aoiPolygon, ideally which has already had no-build areas removed.
    Can take arbitrary polygon definitions in WKT.

    Args:
        dfToFilter (GeoPandas GeoDataFrame): The dataframe to be filtered
        vectorFilePath (str): Path to a vector geometry
        bufferDistance (float): Distance to buffer vectorDF
        removeIntersected (Bool): If True, removes any rows in dfToFilter
            which intersect vectorDF.  If False, removes any rows which do not.

    Returns:
        filteredDF (GeoPandas GeoDataFrame): A subset of dfToFilter

    Raises:
        None

    Todo:
        * Implement units and rotationUnits

    Tests:
        None
    """
    start = datetime.datetime.now()
    vectorDF = gpd.read_file(vectorFilePath)
    returnText = "Retained"
    if removeIntersected:
        filteredDF = dfToFilter[~dfToFilter.intersects(vectorDF.buffer(bufferDistance).unary_union)]
    else:
        returnText = "Retained"
        filteredDF = dfToFilter[dfToFilter.intersects(vectorDF.buffer(bufferDistance).unary_union)]
    end = datetime.datetime.now()
    timeElapsed = end - start
    initialFeatures = len(dfToFilter.index)
    filteredFeatures = len(filteredDF.index)
    print("%s %s of %s candidates in %s seconds" %(returnText,filteredFeatures,initialFeatures,timeElapsed.seconds))
    return filteredDF

def minimumDistanceFromEvaluationToDataFrameFeatures(evaluationDF,vectorDF,columnName='distance'):
        """ Implements Euclidean distance from a data frame of candiate polygons
        to a vector data frame

        Assumes that evaluationDF contains polygons and vectorDF contains vectors

        Args:
            evaluationDF (GeoPandas GeoDataFrame): Each row represents a polygon
                geometry to evaluate Euclidean distances for
            vectorDF (GeoPandas GeoDataFrame): Each row represents a vector
                geometry

        Returns:
            evaluationDF (GeoPandas GeoDataFrame): Appends the minimum distance
                to vectorDF in the column 'distance'

        Raises:
            None

        Tests:Summary line
            None
        """
        try:
            minDistances = []
            for i,row in evaluationDF.iterrows():
                minDistance = vectorDF.distance(row.geometry).min()
                minDistances.append(minDistance)
            evaluationDF[columnName] = minDistances
            return evaluationDF
        except Exception as e:
            print(e)

def buildDistanceFromVectorLayerFromXML(evaluationDF,criteriaRow):
    """ Converts XML into vector distance evaluation

        Evaluation function

    Args:
        evaluationDF (lxml): Set of candidate solutions
        criteriaRow (lxml): filterByVectorBufferDistance

    Returns:
        evaluationDF (GeoPandas GeoDataFrame): Scored and subsetted dataframe
            based upon analysis

    Raises:
        None

    Tests:
        None
    """
    try:
        criteriaName = criteriaRow.attrib['criteriaName']
        layerPath = criteriaRow.attrib['layerPath']
        lowerBound = str(criteriaRow.attrib['lowerBound'])
        upperBound = str(criteriaRow.attrib['upperBound'])

        if lowerBound == "-INF":
            lowerBound = -1.0
        else:
            lowerBound = float(lowerBound)
        if upperBound == "INF":
            upperBound = 100000.0
        else:
            upperBound = float(upperBound)

        vectorDF = gpd.read_file(layerPath)
        vectorQAFNameKludge = "%s" %(criteriaName)
        evaluationDF = minimumDistanceFromEvaluationToDataFrameFeatures(evaluationDF,vectorDF,columnName=vectorQAFNameKludge)

        #evaluationDF = filterByVectorBufferDistance(evaluationDF,layerPath,lowerBound,removeIntersected=True)
        #evaluationDF = filterByVectorBufferDistance(evaluationDF,layerPath,upperBound,removeIntersected=False)
        #evaluationDF[vectorQAFNameKludge] = 100 # KLUDGE, because so expensive to actually calculate

        scores = criteriaRow.find("Scores")
        weight = scores.attrib['weight']
        isZeroExclusionary = scores.attrib['isZeroExclusionary']
        default = scores.attrib['default']
        scoreStructure = []
        for scoreRow in scores:
            lowerBoundInclusive = str(scoreRow.attrib['lowerBoundInclusive'])
            upperBoundExclusive = str(scoreRow.attrib['upperBoundExclusive'])
            score = str(scoreRow.attrib['score'])
            scoreSet = [lowerBoundInclusive,upperBoundExclusive,score]
            scoreStructure.append(scoreSet)
        evaluationDF = candidates.scoreDF(evaluationDF,vectorQAFNameKludge,scoreStructure,isZeroExclusionary=False)
        return evaluationDF
    except Exception as e:
        print(e)

""" Currently does not actually provide a score for vector distance
    scores = criteriaRow.find("Scores")
    weight = scores.attrib['weight']
    isZeroExclusionary = scores.attrib['isZeroExclusionary']
    default = scores.attrib['default']
    scoreStructure = []
    for scoreRow in scores:
        lowerBoundInclusive = str(scoreRow.attrib['lowerBoundInclusive'])
        upperBoundExclusive = str(scoreRow.attrib['upperBoundExclusive'])
        score = str(scoreRow.attrib['score'])
        scoreSet = [lowerBoundInclusive,upperBoundExclusive,score]
        scoreStructure.append(scoreSet)
    evaluationDF = scoreDF(evaluationDF,"totalCutFillVolume",scoreStructure,isZeroExclusionary=False)
"""
