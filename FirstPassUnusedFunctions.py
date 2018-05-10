coding: utf-8 -*-
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

import SpatialIO as io

## HELPFUL FOR DEBUGGING
# %matplotlib inline
# pd.options.display.max_columns = 300

## SETUP
validStats = utils.VALID_STATS

## FUNCTIONS
def filterDataFrameByValue(df,column,argument):
    """ Returns a subset of a GeoPandas GeoDataframe

    Currently only works for single instances of categorical variables.  For more
    complicated cases, either code directly or update this function

    Args:
        df (GeoPandas DataFrame): The dataframe to be filtered
        column (str): The string name of the dataframe column to be filtered
        argument(var): The value determining which rows to return

    Returns:
        filteredDF (GeoPandas DataFrame): A filtered copy of the original df

    Raises:
        None

    Tests:
        None
    """
    filteredDF = df[df[column]==argument]
    return filteredDF

def minimumDistanceFromPointToDataFrameFeatures(x,y,crs,df):
    """ Returns the minimum euclidean distance from a point to the dataframe

    Currently CRS does not project the point

    Args:
        x (float): x-coordinate in the same projection as the dataframe
        y (float): y-coordinate in the same projection as the dataframe
        crs (ENUM CRS): The projection of the point
        df (GeoPandas DataFrame): A dataframe of vector features

    Returns:
        minDistance (float): the smallest euclidean distance calculated

    Raises:
        None

    Tests:
        None
    """
    point = Point(x,y)
    minDistance = df.distance(point).min()
    return minDistance

def minimumDistanceFromPointToDataFrameFeatures(x,y,crs,df):
    """ Returns the minimum euclidean distance from a point to the dataframe

    Currently CRS does not project the point

    Args:
        x (float): x-coordinate in the same projection as the dataframe
        y (float): y-coordinate in the same projection as the dataframe
        crs (ENUM CRS): The projection of the point
        df (GeoPandas DataFrame): A dataframe of vector features

    Returns:
        minDistance (float): the smallest euclidean distance calculated

    Raises:
        None

    Tests:
        None
    """
    point = Point(x,y)
    minDistance = df.distance(point).min()
    return minDistance

def xmlBuildPolygonBuilder(aoiWKT,polygonBuilderXMLElement):
    """ Builds candidate data frame from the elements included in input.xml

    Allows polygon builder to be built from the current version of the input
    XML file

    Args:
        aoiWKT (str): Well-known-text representation of the AOI, currently assumes
            it is passed in WGS1984 (EPSG:4326)
        polygonBuilderXMLElement (lxml.etree._Element): The parameters for
            polygonBuilder

    Returns:
        df (GeoPandas GeoDataFrame): A regularly spaced and rotated candidate
            solution dataframe, in EPSG:3857

    Raises:
        TypeError: If the XML element tag does not reflect that the input
            parameter is for the function polygonBuilder

    Todo:
        * Implement ability to project

    Tests:
        None
    """
    if (polygonBuilderXMLElement.tag != "PolygonBuilder"):
        raise TypeError('Invalid Type','Expected Parameters for Polygon Builder')
    wkt = polygonBuilderXMLElement.attrib['wkt']
    gridSpacing = float(polygonBuilderXMLElement.attrib['gridSpacing'])
    units = polygonBuilderXMLElement.attrib['units']
    rotationStart = float(polygonBuilderXMLElement.attrib['rotationStart'])
    rotationStop = float(polygonBuilderXMLElement.attrib['rotationStop'])
    rotationUnits = polygonBuilderXMLElement.attrib['rotationUnits']
    rotationSpacing = float(polygonBuilderXMLElement.attrib['rotationSpacing'])
    wktPolygon = wktToShapelyPolygon(aoiWKT,4326,to_epsg=3857)
    df = polygonBuilder(wktPolygon,wkt=wkt,units=units,gridSpacing=gridSpacing,
                        rotationUnits=rotationUnits,rotationStart=rotationStart,
                        rotationStop=rotationStop,rotationSpacing=rotationSpacing)
    return df

def convertRasterToNumpyArray(raster_path):
    """ Generates a numpy array from a GeoTiff


    Args:
        raster_path (str): Filepath of a GeoTiff

    Returns:
        data (Numpy Array): The first band of the raster, as an array

    Raises:
        None

    Tests:
        >>> raster_path = "./test_data/testelevunproj.tif"
        >>> convertRasterToNumpyArray(raster_path)
    """


    dataset = gdal.Open(raster_path)
    band = dataset.GetRasterBand(1)
    geotransform = dataset.GetGeoTransform()
    xinit = geotransform[0]
    yinit = geotransform[3]

    xsize = geotransform[1]
    ysize = geotransform[5]
    data = band.ReadAsArray() # whole band
    # data = band.ReadAsArray(col1, row1, col2 - col1 + 1, row2 - row1 + 1) # partial band
    print data.shape
    print np.mean(data)
    return data

def queryRasterValueForPoint(x,y,raster_path,pointCRS=None,rasterCRS=None):
    """ Returns the value of a GeoTiff at a specified point

    pointCRS and rasterCRS are not yet implemented, and therefore both should
    be in the same coordinate system

    Args:
        x (float): The x-coordinate, in the same CRS as the raster
        y (float): The y-coordinate, in the same CRS as the raster
        raster_path (str): Filepath to a GeoTiff
        pointCRS (ENUM CRS): Not implemented
        rasterCRS (ENUM CRS): Not implemented

    Returns:
        pointQuery (float): The value of the raster at location (x,y)

    Raises:
        None

    Todo:
        * Implement projections

    Tests:
        None
    """
    point = "POINT(%s %s)" %(x,y)
    pointQuery = point_query(point,raster_path)
    return pointQuery

def convertSubsettedEvaluationDFIntoPolygonGrid(evaluationDF, squareDimension):
    polygonEvaluations = []
    oldPolygons = []
    for geometry in evaluationDF.geometry:
        oldPolygons.append(geometry)
        centroid_x = geometry.centroid.x
        centroid_y = geometry.centroid.y
        offset = squareDimension / 2
        square = Polygon([[centroid_x - offset,centroid_y - offset],
                          [centroid_x + offset,centroid_y - offset],
                          [centroid_x + offset,centroid_y + offset],
                          [centroid_x - offset,centroid_y + offset]])
        polygonEvaluations.append(square)
    evaluationDF = evaluationDF.drop('geometry',axis=1)
    evaluationDF['geometry'] = polygonEvaluations
    evaluationDF['old_geo'] = oldPolygons
    return evaluationDF

def generateRandomLatLonPair(latMin,latMax,lonMin,lonMax):
    """ Creates random points to help test other functions

    This code is a helper function

    Args:
        latMin (float): Lower bound of latitude
        latMax (float): Upper bound of latitude
        lonMin (float): Lower bound of longitude
        lonMax (float): Upper bound of longitude

    Returns:
        lat (float): A random latitude within the specified range
        lon (float): A random longitude within the specified range

    Raises:
        None

    Tests:
        None
    """
    lat = np.random.uniform(latMin,latMax)
    lon = np.random.uniform(lonMin,lonMax)
    return lat,lon

def generateRandomCandidateDataFrame(nCandidates,latMin,latMax,lonMin,lonMax):
        """ Creates random solutionDF to help test other functions

        This code is a helper function

        Args:
            nCandidates (int): Number of candidate sites to create
            latMin (float): Lower bound of latitude
            latMax (float): Upper bound of latitude
            lonMin (float): Lower bound of longitude
            lonMax (float): Upper bound of longitude

        Returns:
            lat (float): A random latitude within the specified range
            lon (float): A random longitude within the specified range

        Raises:
            None

        Tests:
            None
        """
        lats = []
        lons = []
        scores = []
        geoms = []
        for i in range(0,nCandidates):
            lat,lon = generateRandomLatLonPair(latMin,latMax,lonMin,lonMax)
            lats.append(lat)
            lons.append(lon)
            score = np.random.randint(0,101)
            scores.append(score)
            point = Point([lon,lat])
            square = Polygon([[x,y],[x+gridSpacing,y],[x+gridSpacing,y+gridSpacing],[x,y+gridSpacing]])
            if square.within(polygon):
                squareList.append(square)
        candidateDF = gpd.GeoDataFrame({""})
        return 0

def buildTestSet(xmlPath):
    tree = ET.parse(xmlPath)
    root = tree.getroot()

    resultDir = root.attrib['resultDir']
    studyID = root.attrib['studyID']
    epsg = root.attrib['epsg']
    print "%s %s %s" %(resultDir,studyID,epsg)

    siteSearches = root.find("SiteSearches")
    layerIDs = []
    evaluationDFs = []
    for siteSearch in siteSearches:
        siteSearch_studyObjectiveID = siteSearch.attrib['studyObjectiveID']
        siteSearch_layerID = siteSearch.attrib['layerID']
        siteSearch_type = siteSearch.attrib['type']
        siteSearch_name = siteSearch.attrib['name']
        siteSearch_note = siteSearch.attrib['note']
        siteSearch_nReturn = siteSearch.attrib['nReturn']

        siteConfiguration = siteSearch.find("SiteConfiguration")[0]
        if siteConfiguration.tag == "WKTTemplate":
            print "WKT Template"

        searchParameters = siteSearch.find("SearchParameters")[0]
        if searchParameters.tag == "GriddedSearch":
            evaluationDF = buildGriddedSearchFromXML(siteConfiguration,searchParameters)

        siteEvaluation = siteSearch.find("SiteEvaluation")
        weights = []
        qafNames = []
        scoringDict = {}
        criteriaCount = 0
        for criteriaRow in siteEvaluation:
            criteriaCount += 1
            # set the column name if none
            if criteriaRow.attrib["criteriaName"] == "None":
                criteriaRow.attrib["criteriaName"] = "Criteria_%s" %(criteriaCount)

            # Get the metadata needed for scoring

        evaluationDFs.append(evaluationDF)


    return evaluationDFs,siteSearches
