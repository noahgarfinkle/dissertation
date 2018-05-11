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

""" REFERENCE CODE
setting up blank raster: dst_ds.SetGeoTransform([topLeftX,pixel_width,0,topLeftY,0,-pixel_height])
"""


## FUNCTION
def filterDataFrameByBounds(df,lx,ly,ux,uy,bufferDistance=0):
    """ Subsets a GeoPandas DataFrame by bounding box

    Very useful for shrinking the number of features.  Currently assumes that the bounding
    box and optional buffer distance is in the same projection as the dataframe.

    Args:
        df (GeoPandas DataFrame): The dataframe to be filtered
        lx (float): Lower left x coordinate of the bounding box
        ly (float): Lower left y coordinate of the bounding box
        ux (float): Upper right x coordinate of the bounding box
        uy (float): Upper right y coordinate of the bounding box
        bufferDistance (float): Optional distance for buffering the bounding box

    Returns:
        filteredDF (GeoPandas DataFrame): A filtered copy of the original df

    Raises:
        None

    Tests:
        None
    """
    filteredDF = df.cx[lx-bufferDistance:ux+bufferDistance,ly-bufferDistance:uy+bufferDistance]
    return filteredDF

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

def wktToShapelyPolygon(wkt,epsg,to_epsg=None):
    """ Creates a Shapely polygon from wkt, and projects if specified

    Args:
        wkt (str): A string of well-known-text
        epsg (int): The current projection of wkt
        to_epsg (int): The desired projection of wkt, not used if None

    Returns:
        wktPolygon (Shapely Geometry): Shapely representation of wkt

    Raises:
        None

    Tests:
        None
    """
    wktPolygon = loads(wkt)
    if to_epsg:
        df_to_project = gpd.GeoDataFrame([wktPolygon])
        df_to_project.columns = ['geometry']
        df_to_project.crs = {'init':'EPSG:' + str(epsg)}
        df_to_project = df_to_project.to_crs({'init':'EPSG:' + str(to_epsg)})
        wktPolygon = df_to_project.geometry[0]
    return wktPolygon

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


# First pass implementation for class
"""
aoiDF = gpd.read_file("../FLW_Missouri Mission Folder/SUPPORT/Staging.shp")
aoiDF = aoiDF.to_crs({'init':'epsg:3857'})
squareDimension = 400
aoiDF.plot()

# Airfield Objective
airfieldAOI = aoiDF[aoiDF['Stage']=='Gold'].reset_index().geometry[0]
airfieldEvaluationDataFrame = generateEvaluationGridDataFrame(airfieldAOI,100)
slopePath = '../FLW_Missouri Mission Folder/RASTER/slope_proj.tif'
airfieldSlopeEvaluationDataFrame = generateRasterStatisticsForDataFrame(airfieldEvaluationDataFrame,slopePath,stats="mean max",isCategorical=False)
airfieldSlopeEvaluationDataFrame.head()
airfieldSlopeEvaluationDataFrame.plot(column='max')
plt.hist(airfieldSlopeEvaluationDataFrame['max'])
airfieldSlopeEvaluationDataFrameSubset = airfieldSlopeEvaluationDataFrame[airfieldSlopeEvaluationDataFrame['max'] < 2]
airfieldSlopeEvaluationDataFrameSubset.head()

len(airfieldSlopeEvaluationDataFrameSubset.index)
airfieldSlopeEvaluationDataFrameSubset.plot(column='max')


largerAirfields = convertSubsettedEvaluationDFIntoPolygonGrid(airfieldSlopeEvaluationDataFrameSubset, 800)
largerAirfields.head()
largerAirfields.plot()

elevationPath = "../FLW_Missouri Mission Folder/RASTER/DEM_CMB_ELV_SRTMVF2_proj.tif"
largerAirfields.head()
airfieldEvaluationDataFrame.head()

cutFillDF = calculateCutFill(largerAirfields,elevationPath,finalElevation='mean',rasterResolution=30)
cutFillDF.head()
cutFillDF.plot(column='totalCutFillVolume')

plt.hist(cutFillDF['totalCutFillVolume'])
"""
"""
# Base Objective 1
baseObjective1AOI = aoiDF[aoiDF['Stage']=='Gold'].reset_index().geometry[0]
baseObjective1EvaluationDataFrame = generateEvaluationGridDataFrame(baseObjective1AOI,100)


# Base Objective 2
baseObjective2AOI = aoiDF[aoiDF['Stage']=='Gold'].reset_index().geometry[0]
baseObjective2EvaluationDataFrame = generateEvaluationGridDataFrame(baseObjective2AOI,100)
"""






"""
raster_path = "../FLW_Missouri Mission Folder/RASTER/DEM_CMB_ELV_SRTMVF2_proj.tif"
result_DF = generateRasterStatisticsForDataFrame(evaluationGridDataFrame,raster_path,stats="mean",isCategorical=False)
plt.figure()
plt.suptitle("Plot of mean elevation for AOI")
result_DF.plot(column='mean')
plt.savefig("./results/meanelevationforAOI.png")
evaluationGridDataFrame.plot()

start = datetime.datetime.now()
cutFillDF = calculateCutFill(evaluationGridDataFrame,raster_path,finalElevation='mean',rasterResolution=30)
end = datetime.datetime.now()
elapsedTime = end - start
nItems = len(cutFillDF.index)
print "Evaluated %s features in %s seconds" %(nItems,elapsedTime.seconds)
plt.figure()
plt.suptitle("Cut fill for evaluation cells (m3) for AOI")
array = cutFillDF['mini_raster_array'][0]
array
plt.imshow(array)
cutFillDF.head()
cutFillDF.plot(column="totalCutFillVolume")
plt.savefig("./results/cutfillquantity.png")

cutFillDF.head()

plt.figure()
plt.suptitle("Cut fill for evaluation cells (m3) for AOI")
plt.xlabel("Total volume cut/fill required (m3)")
plt.ylabel("Number of evaluation sites")
plt.hist(cutFillDF['totalCutFillVolume'])
plt.savefig("./results/cutfillhistogram.png")



meaninglessCategoricalDF = generateRasterStatisticsForDataFrame(df_subset,raster_path,isCategorical=True)
meaninglessCategoricalDF

# minimum distance from each raster cell to roads


vector_path = "./test_data/UtilityInfrastructureCrv_3.shp"
roadsDF = gpd.read_file(vector_path)
roadsDF.crs = {'init':'epsg:3857'}
roadsDF = roadsDF.to_crs({'init':'epsg:3857'})
evaluationDF = evaluationGridDataFrame
vectorDF = roadsDF
lx,ly,ux,uy = evaluationDF.total_bounds

vectorDF_filtered = filterDataFrameByBounds(vectorDF,lx,ly,ux,uy)
vectorDF_filtered.plot()

minDistances = []
for i,row in evaluationDF.iterrows():
    minDistance = vectorDF_filtered.distance(row.geometry).min()
    minDistances.append(minDistance)
evaluationDF['distance'] = minDistances
plt.figure()
f,ax = plt.subplots(1)
f.suptitle("Minimum distance from infrastructure")
evaluationDF.plot(column='distance',ax=vectorDF_filtered.plot())
plt.savefig("./results/distancefrominfrastructure.png")



a = evaluationDF[0:1]
a
a.to_html()
"""

# FIRST PASS IMPLEMENTATION
def testFirstPassImplementation():
    xmlPath = "./input.xml"
    raster_path = "../FLW_Missouri Mission Folder/RASTER/DEM_CMB_ELV_SRTMVF2.tif"
    vector_path = "./test_data/UtilityInfrastructureCrv_3.shp"
    df = gpd.read_file('./test_data/MO_2016_TIGER_Counties_shp/MO_2016_TIGER_Counties_shp.shp')
    df.CRS = {'init':'epsg:4326'}
    df_proj = df.to_crs({'init':'epsg:3857'})
    aoiWKT = df_proj[df_proj["NAME"] == "Pulaski"].geometry.values[0].to_wkt()
    aoiLoaded = loads(aoiWKT)

    roadsDF = gpd.read_file(vector_path)
    roadsDF.crs = {'init':'epsg:3857'}
    roadsDF = roadsDF.to_crs({'init':'epsg:3857'})

    lx,ly,ux,uy = df_proj['geometry'][4].envelope.bounds
    filteredRoads = filterDataFrameByBounds(roadsDF,lx,ly,ux,uy)
    filteredCounties = filterDataFrameByBounds(df_proj,lx,ly,ux,uy)
    filteredCounties.plot(ax=filteredRoads.plot(facecolor='red'),facecolor='Green')

def testPointDistance():
    point = Point(ux,uy)
    print roadsDF.distance(point).min()

def createTestGrid():
    aoiLoaded = loads(aoiWKT)
    testList = generateEvaluationGridDataFrame(aoiLoaded,3200)
    testList.plot()

# test.zonal_stats
def test_zonal_stats():
    raster_path = "./test_data/testelevunproj.tif"
    dfStatsCategorical = generateRasterStatisticsForDataFrame(df,raster_path,isCategorical=True)
    dfStatsCategorical.head()
    dfStatsCategorical.plot(column="mean")

    dfStatsNonCategorical = generateRasterStatisticsForDataFrame(df,raster_path,isCategorical=False)
    dfStatsNonCategorical.head()
    dfStatsNonCategorical.plot(column='majority')

# raster tests
def test_rasterStatCroppedRaster(index = 40):
    df_subset = df[index:index + 1]
    rasterDF = rasterStatCroppedRaster(df_subset,raster_path)
    masked_array = rasterDF['mini_raster_array'][0]
    masked_array_np_masked = ma.masked_array(masked_array)
    masked_array_np = np.array(masked_array)
    plt.figure()
    plt.imshow(masked_array_np_masked)
    plt.figure()
    plt.imshow(masked_array_np)
    print "Mean Unmasked: %s, Mean Masked: %s" %(np.mean(masked_array_np),np.mean(masked_array_np_masked))
    np.mean(masked_array_np)
    np.mean(masked_array_np_masked)
    return df['geometry'][index]



# cut fill tests
def testCutFill():
    index = 40
    df_subset = df[index:index + 2]
    df_subset = df_subset.reset_index()
    dem_path = raster_path
    finalElevation = 'mean'

    appendedDF = calculateCutFill(df_subset,raster_path)
    appendedDF
    appendedDF.plot()
    plt.imshow(appendedDF['elevationChangeArray'][1])
