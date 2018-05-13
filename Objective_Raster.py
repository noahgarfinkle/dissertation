# -*- coding: utf-8 -*-
"""
Manages simple objective functions for raster data layers
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
import Objective_Vector as objective_vector
import pgdissroute as pgdissroute
import SpatialIO as io
import SpatialOpt as opt

## RASTER OPERATIONS
def generateRasterStatisticsForDataFrame(df,raster_path,stats="count majority minority unique mean",
                                            colName="slope",isCategorical=False):
    """ Produces neighborhood statistics for a raster based on each feature in a
        GeoPandas GeoDataFrame

    Works for both categorical and continuous rasters.  Note that some stats, such
    as count, substantially increase processing time.

    Args:
        df (GeoPandas GeoDataFrame): Each row represents a polygon geometry to
            generate neighborhood statistics for
        raster_path (str): Filepath of a GeoTiff raster
        stats (str): space-separated list of valid statistics from validStats
        colName (str): What to prepend to the statistic column in the new dataframe,
            done in order to avoid repeated column names
        isCategorical (bool): Raster values will be interpreted as categorical if
            True, continuous if False.  Defaults False.

    Returns:
        newDF (GeoPandas GeoDataFrame): The input df, merged with the appropriate
            raster statistics

    Raises:
        None

    Tests:
        None
    """
    try:
        row_stats_df = gpd.GeoDataFrame(raster_stats(vectors=df['geometry'],raster=raster_path,stats=stats, copy_properties=True, nodata_value=0, categorical=isCategorical))
        row_stats_df.index = df.index

        # rename the columns
        for stat in stats.split(' '):
            newColName = "%s_%s" %(colName,stat)
            row_stats_df.rename(columns={stat:newColName}, inplace=True)
        # rename any remaining columns, such as those created using count
        # this can be accomplished because the only columns should be geometry or colName_preceeded
        columnsToDrop = []
        for columnName in row_stats_df.columns:
            originalName = columnName
            columnName = str(columnName)
            if columnName == "geometry" or colName in columnName:
                pass
            else:
                newColName = "%s_%s" %(colName,columnName)
                columnsToDrop.append(newColName)
                row_stats_df.rename(columns={originalName:newColName}, inplace=True)
        newDF = gpd.GeoDataFrame(pd.concat([df,row_stats_df],axis=1))
        return newDF,columnsToDrop
    except Exception as e:
        print e

def rasterStatCroppedRaster(df,raster_path):
    """ Produces neighborhood statistics for a raster based on each feature in a
        GeoPandas GeoDataFrame, also returning a column with the raster values for
        each row

    This implementation of zonal statistics is very useful for functions such as
    cut-fill, returning numpy masked arrays of each subraster inside the
    GeoDataFrame.  Note this function currently uses the default values for
    zonal_stats, and should in the future be merged with rasterStat, with a flag
    value parameter for returning the sub-rasters

    Args:
        df (GeoPandas GeoDataFrame): Each row represents a polygon geometry to
            generate neighborhood statistics for
        raster_path (str): Filepath of a GeoTiff raster

    Returns:
        newDF (GeoPandas GeoDataFrame): A data frame with each sub-raster contained
        in the column 'mini_raster_array'

    Raises:
        None

    Tests:
        None
    """
    try:
        rasterSource = zonal_stats(df['geometry'],raster_path,all_touched=True,raster_out=True)
        rasterDF = pd.DataFrame(rasterSource)
        return rasterDF
    except Exception as e:
        print e

def buildCategoricalRasterStatFromXML(evaluationDF,criteriaRow):
    """ Converts XML into categorical raster statistic evaluation

        Evaluation function

    Args:
        evaluationDF (lxml): Set of candidate solutions
        criteriaRow (lxml): CategoricalRasterStat

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
        start = datetime.datetime.now()


        if lowerBound == "-INF":
            lowerBound = -1.0
        else:
            lowerBound = float(lowerBound)
        if upperBound == "INF":
            upperBound = 101.0
        else:
            upperBound = float(upperBound)

        valueList = criteriaRow.attrib['valueList']


        initialDataFrameSize = len(evaluationDF.index)

        evaluationDF,columnsToDrop = generateRasterStatisticsForDataFrame(evaluationDF,layerPath,stats="count",colName=criteriaName,isCategorical=True)
        end_EvaluationDF = datetime.datetime.now()
        timeElapsed = end_EvaluationDF - start
        # replace NA values with zero, note this may need to be moved into the first pass function to make sure I do not unintentionally overwrite other data
        evaluationDF = evaluationDF.fillna(0)
        end_fillna = datetime.datetime.now()
        timeElapsed = end_fillna - end_EvaluationDF
        # calculate percentages
        values = valueList.split(',')
        totalCountColumnName = "%s_count" %(criteriaName)
        countColumnNames = []
        for value in values:
            countColumnName = "%s_%s" %(criteriaName,value)
            countColumnNames.append(countColumnName)

        evaluationDF[criteriaName] = 0
        for countColumnName in countColumnNames:
            if evaluationDF.columns.contains(countColumnName):
                evaluationDF[criteriaName] += evaluationDF[countColumnName]

        evaluationDF[criteriaName] = evaluationDF[criteriaName] / evaluationDF[totalCountColumnName] * 100.0
        end_CreatingCriteria = datetime.datetime.now()
        timeElapsed = end_CreatingCriteria - end_fillna

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
        evaluationDF = candidates.scoreDF(evaluationDF,criteriaName,scoreStructure,isZeroExclusionary=isZeroExclusionary)
        # remove the count columns
        columnsToDrop.append(totalCountColumnName)
        evaluationDF = evaluationDF.drop(columnsToDrop,axis=1)
        end_scoring = datetime.datetime.now()
        timeElapsed = end_scoring - end_CreatingCriteria
        return evaluationDF
    except Exception as e:
        print e

def buildContinuousRasterStatFromXML(evaluationDF,criteriaRow):
    """ Converts XML into continuous raster statistic evaluation

        Evaluation function

    Args:
        evaluationDF (lxml): Set of candidate solutions
        criteriaRow (lxml): ContinuousRasterStat

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

        stat = criteriaRow.attrib['stat']
        initialDataFrameSize = len(evaluationDF.index)

        evaluationDF,columnsToDrop = generateRasterStatisticsForDataFrame(evaluationDF,layerPath,stats=stat,colName=criteriaName,isCategorical=False)
        evaluationColumnName = "%s_%s" %(criteriaName,stat)
        # KLUDGE- copy the evaluation column
        evaluationDF[criteriaName] = evaluationDF[evaluationColumnName] # in the future replace instead of copy

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
        evaluationDF = candidates.scoreDF(evaluationDF,criteriaName,scoreStructure,isZeroExclusionary=isZeroExclusionary)
        columnsToDrop.append(evaluationColumnName)
        evaluationDF = evaluationDF.drop(columnsToDrop,axis=1)
        return evaluationDF
    except Exception as e:
        print e
