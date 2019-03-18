# -*- coding: utf-8 -*-
"""
Manages custom analytic objective functions
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
import Objective_Raster as objective_raster
import Objective_Vector as objective_vector
# import pgdissroute as pgdissroute
import SpatialIO as io
import SpatialOpt as opt

## CUT-FILL OPERATIONS
def calculateCutFill(df,dem_path,finalElevation='mean',rasterResolution=10):
    """ Generates cut fill values based on a raster

    Need to verify that this function calculates correctly. Additionally, should
    determine raster resolution on it's own, rather than an input.  In the future,
    should return cut and fill values separately.

    Args:
        df (GeoPandas GeoDataFrame): Each row represents a polygon geometry to
            generate cut/fill values for
        raster_path (str): Filepath of a GeoTiff raster containing a digital
        elevation model in the same coordinate system as the desired results
        finalElevation (str/float): must either be number or string from validStats,
            this represents the target elevation to determine cut/fill values for
            at each pixel
        rasterResolution (float): Cut/fill volume is determined by a height change
            multiplied by the square of this number, representing the pixel cell
            size in target units of the DEM

    Returns:
        appendedDF (GeoPandas GeoDataFrame): A data frame with each sub-raster contained
        in the column 'mini_raster_array', along with masked arrays of elevation change
        in 'elevationChangeArray' and the total cut fill volume in totalCutFillVolume

    Raises:
        None

    Tests:Summary line
        None
            if 'old_geo' in df.columns:
                df = df.drop('old_geo',axis=1)

            if 'old_geometry' in df.columns:
                df = df.drop('old_geometry',axis=1)


                    #for i in range(0,len(appendedDF.index+1)):
                        #row = appendedDF[i:i+1]
    """
    # KLUDGE to remove raster stats with the same name
    if finalElevation in df.columns:
        df = df.drop(finalElevation,axis=1)
    df = df.reset_index()
    croppedRasterDF = objective_raster.rasterStatCroppedRaster(df,dem_path)
    appendedDF = gpd.GeoDataFrame(pd.concat([df,croppedRasterDF],axis=1))
    elevationChangeArrays = []
    totalRequiredHeightChanges = []
    totalCutFillVolumes = []
    for i,row in appendedDF.iterrows():
        #maskedRaster = row['mini_raster_array']
        maskedRaster = croppedRasterDF['mini_raster_array'][i] # KLUDGE
        maskedRaster_Array = ma.masked_array(maskedRaster)
        targetElevation = -999
        if isinstance(finalElevation,basestring):
            targetElevation = row[finalElevation]
        else:
            targetElevation = finalElevation
        requiredHeightChange = np.subtract(maskedRaster_Array,targetElevation)
        totalRequiredHeightChange = np.sum(np.abs(requiredHeightChange))
        totalCutFillVolume = totalRequiredHeightChange * rasterResolution * rasterResolution
        elevationChangeArrays.append(requiredHeightChange)
        totalCutFillVolumes.append(totalCutFillVolume)

    #appendedDF['elevationChangeArray'] = elevationChangeArrays
    appendedDF['totalCutFillVolume'] = totalCutFillVolumes
    return appendedDF

def buildCutFillFromXML(evaluationDF,criteriaRow):
    """ Converts XML into cut/fill evaluation

        Evaluation function

    Args:
        evaluationDF (lxml): Set of candidate solutions
        criteriaRow (lxml): CutFill

    Returns:
        evaluationDF (GeoPandas GeoDataFrame): Scored and subsetted dataframe
            based upon analysis

    Raises:
        None

    Tests:
        None
    """
    criteriaName = criteriaRow.attrib['criteriaName']
    layerPath = criteriaRow.attrib['layerPath']
    lowerBound = str(criteriaRow.attrib['lowerBound'])
    upperBound = str(criteriaRow.attrib['upperBound'])

    if lowerBound == "-INF":
        lowerBound = -1.0
    else:
        lowerBound = float(lowerBound)
    if upperBound == "INF":
        upperBound = 10000000.0
    else:
        upperBound = float(upperBound)

    evaluationDF = calculateCutFill(evaluationDF,layerPath,finalElevation='mean',rasterResolution=1)
    evaluationDF[criteriaName] = evaluationDF["totalCutFillVolume"] # KLUDGE, in the future replace the column name
    initialNumber = len(evaluationDF.index)
    #evaluationDF = evaluationDF[evaluationDF["totalCutFillVolume"] < upperBound]
    finalNumber = len(evaluationDF.index)
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
    evaluationDF = candidates.scoreDF(evaluationDF,criteriaName,scoreStructure)
    columnsToDrop = ['mini_raster_affine','mini_raster_array','mini_raster_nodata','count','max','mean','min','totalCutFillVolume']
    evaluationDF = evaluationDF.drop(columnsToDrop,axis=1)
    return evaluationDF
