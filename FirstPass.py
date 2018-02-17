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
from rasterstats import zonal_stats, raster_stats, point_query, utils
import matplotlib.pyplot as plt

import SpatialIO as io

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

""" REFERENCE CODE
setting up blank raster: dst_ds.SetGeoTransform([topLeftX,pixel_width,0,topLeftY,0,-pixel_height])
"""

## CLASSES
class SiteSearch:
    """ Stores the individual study objective

        Corresponds to a study objective in ENSITE, such as a contingency base or an
        airfield

        Attributes:
            crs (Enum): The projection for the study
            AreaOfInterest (Shapely polygon): The area of the overall study objective
    """

    def __init__(self):
        self.crs = None
        self.AreaOfInterest = None

class SiteSuitabilityCriteria:
    """ An individual geospatial evaluation

        Sets of site suitability criteria and scores comprise a site search

        Attributes:
            None
    """

    def __init__(self):
        return None


class SiteRelationalConstraint:
    """ Encodes the relationship between two study objectives

        For instance, routing distance or topological relationships between study
        objectives

        Attributes:
            None
    """

    def __init__(self):
        return None

class Input:
    """ Python data structure for parsing the XML data structure

        Attributes and methods for parsing the XML input file, ensuring that first
        pass is usable across a range of applications

        Attributes:
            xmlPath (str): Path to the XML file
            tree (XML tree): Python representation of the XML file using lxml etree
            root (XML tree element): Root node of the lxml etree
            siteSearches (list: SiteSearch): The study objectives
            siteRelationalConstraints (list: SiteRelationalConstraint): The relationships
                between the study objectives
            resultDir (str): Base path for writing results to
            studyObjectiveID (long): Study objectiveve for reference to PostGIS database
                object
    """

    def __init__(self,xmlPath):
        self.xmlPath = xmlPath
        self.tree = ET.parse(xmlPath)
        self.root = tree.getroot()
        self.siteSearches = self.retrieveSiteSearches()
        self.siteRelationalConstraints = self.retrieveSiteRelationalConstraints()
        self.resultDir = root.attrib['resultDir']
        self.studyObjectiveID = root.attrib['studyObjectiveID']

    def pretty_print(self):
        print ET.tostring(self.root,pretty_print=True)

    def retrieveSiteSearches(self):
        """ Converts each XML study objective to type SiteSearch

        Retreives each indivdidual study objective, such as an individual airfield, and
        appends to a list

        Args:
            None

        Returns:
            None (updates siteSearches)

        Raises:
            None

        Tests:
            None
        """
        siteSearches = []
        siteSearchesElement = self.root[0]
        for siteSearchElement in siteSearchesElement:
            print "%s: %s" %(siteSearchElement.tag, siteSearchElement.attrib)
            for siteSearchElementChild in siteSearchElement:
                print "\t%s: %s" %(siteSearchElementChild.tag,siteSearchElementChild.attrib)
            siteSearch = SiteSearch()
            siteSearches.append(siteSearch)
        return siteSearches

    def retrieveSiteRelationalConstraints(self):
        """ Converts each XML relational constraint to type SiteRelationalConstraint

        Captures the interelationships between study objectives, linking the objects

        Args:
            None

        Returns:
            None (updates siteRelationalConstraints)

        Raises:
            None

        Tests:
            None
        """
        siteRelationalConstraints = self.root[1]
        siteRelationalConstraint = SiteRelationalConstraint()
        siteSearchRelationalConstraints = []
        siteRelationalConstraints.append(siteRelationalConstraint)
        return siteSearchRelationalConstraints

## FUNCTIONS
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

def projectWKT(wkt,from_epsg,to_epsg):
    """ Reprojects a string of well known text


    Args:
        wkt (str): A string of well-known-text
        from_epsg (int): The current projection of wkt
        to_epsg (int): The desired projection of wkt

    Returns:
        reprojectedWKT (str): well-known-text projected into to_epsg

    Raises:
        None

    Tests:
        None
    """
    feat = loads(wkt)
    df_to_project = gpd.GeoDataFrame([feat])
    df_to_project.columns = ['geometry']
    df_to_project.crs = {'init':'EPSG:' + str(from_epsg)}
    df_to_project.to_crs({'init':'EPSG:' + str(to_epsg)})
    reprojectedWKT = df_to_project.geometry[0].to_wkt()
    return reprojectedWKT

def createEmptyRaster(rasterPath,topLeftX,topLeftY,cellSize,width,height,epsg):
    """ Generates a raster failled with zeros in GeoTiff format

    Uses GDAL to create an empty raster

    Args:
        rasterPath (str): Where to place the newly created raster
        topLeftX (float): The x-coordinate of the top left of the raster, in raster
            projection
        topLeftY (float): The y-coordinate of the top left of the raster, in raster
            projection
        cellSize (float): The raster resolution, assumes that the raster cell
            size is the same in the x and y dimension
        width (int): X-dimension of the raster, in pixels
        height (int): Y-dimension of the raster, in pixels
        epsg (int): The EPSG representation of the projection of the raster

    Returns:
        rasterPath (str): The path of the created raster, as a confirmation that
            the raster was written

    Raises:
        None

    Tests:
        <<< raster[25:50,25:50] = 100 # this code is for testing, and not a unit test
    """
    geotransform = [topLeftX,cellSize,0,topLeftY,0,-cellSize]
    driver = gdal.GetDriverByName("GTiff")
    dst_ds = driver.Create(rasterPath, width, height, 1, gdal.GDT_Byte )
    dst_ds.SetGeoTransform(geotransform)
    srs = osr.SpatialReference()
    srs.ImportFromEPSG(epsg)
    dst_ds.SetProjection(srs.ExportToWkt())
    raster = np.zeros((height,width),dtype=np.uint32)
    dst_ds.GetRasterBand(1).WriteArray(raster)
    return rasterPath


def floatrange(start, stop, step):
    """ Generates a range between two floats with a float step size

    Taken from http://portolan.leaffan.net/creating-sample-points-with-ogr-and-shapely-pt-2-regular-grid-sampling/

    Args:
        start (float): Lower bound, inclusive
        stop (float): Upper bound, exclusive
        step (float): Step size

    Returns:
        generator (generator of floats): Range between start and stop

    Raises:
        None

    Tests:
        None
    """
    while start < stop:
        yield start
        start += step

def generateEvaluationGridDataFrame(polygon,gridSpacing):
    """ Summary line

    Detailed description

    Args:
        param1 (int): The first parameter.
        param1 (str): The second parameter.

    Returns:
        network (pandas dataframe): The return and how to interpret it

    Raises:
        IOError: An error occured accessing the database

    Tests:
        >>> get_nearest_node(-92.1647,37.7252)
        node_id = 634267, dist = 124
    """
    pointList = []
    bounds = polygon.bounds
    ll = bounds[:2]
    ur = bounds[2:]
    for x in floatrange(ll[0],ur[0],gridSpacing):
        for y in floatrange(ll[1],ur[1],gridSpacing):
            point = Point(x,y)
            if point.within(polygon):
                pointList.append(point)
    evaluationGridDataFrame = gpd.GeoDataFrame(pointList)
    evaluationGridDataFrame.columns = ['geometry',score]
    return evaluationGridDataFrame

def convertRasterToNumpyArray(raster_path):
    """ Summary line

    Detailed description

    Args:
        param1 (int): The first parameter.
        param1 (str): The second parameter.

    Returns:
        network (pandas dataframe): The return and how to interpret it

    Raises:
        IOError: An error occured accessing the database

    Tests:
        >>> get_nearest_node(-92.1647,37.7252)
        node_id = 634267, dist = 124
    """
    raster_path = "./test_data/testelevunproj.tif"

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

def generateRasterStatisticsForDataFrame(df,raster_Path,stats="count majority minority unique mean",isCategorical=False):
    """ Summary line

    Detailed description

    Args:
        param1 (int): The first parameter.
        param1 (str): The second parameter.

    Returns:
        network (pandas dataframe): The return and how to interpret it

    Raises:
        IOError: An error occured accessing the database

    Tests:
        >>> get_nearest_node(-92.1647,37.7252)
        node_id = 634267, dist = 124
    """
    row_stats_df = gpd.GeoDataFrame(raster_stats(vectors=df['geometry'],raster=raster_path,stats=stats, copy_properties=True, nodata_value=0, categorical=isCategorical))
    newDF = gpd.GeoDataFrame(pd.concat([df,row_stats_df],axis=1))
    return newDF

# Note: Requires point in same coordinate system as raster, I assume
# todo: implement projections
def queryRasterValueForPoint(x,y,raster_path,pointCRS=None,rasterCRS=None):
    """ Summary line

    Detailed description

    Args:
        param1 (int): The first parameter.
        param1 (str): The second parameter.

    Returns:
        network (pandas dataframe): The return and how to interpret it

    Raises:
        IOError: An error occured accessing the database

    Tests:
        >>> get_nearest_node(-92.1647,37.7252)
        node_id = 634267, dist = 124
    """
    point = "POINT(%s %s)" %(x,y)
    return point_query(point,raster_path)

def rasterStatCroppedRaster(df,raster_path):
    """ Summary line

    Detailed description

    Args:
        param1 (int): The first parameter.
        param1 (str): The second parameter.

    Returns:
        network (pandas dataframe): The return and how to interpret it

    Raises:
        IOError: An error occured accessing the database

    Tests:
        >>> get_nearest_node(-92.1647,37.7252)
        node_id = 634267, dist = 124
    """
    rasterSource = zonal_stats(df['geometry'],raster_path,all_touched=True,raster_out=True)
    rasterDF = pd.DataFrame(rasterSource)
    return rasterDF

#finalElevation must either be number or string from validStats
def calculateCutFill(df,dem_path,finalElevation='mean',rasterResolution=10):
    """ Summary line

    Detailed description

    Args:
        param1 (int): The first parameter.
        param1 (str): The second parameter.

    Returns:
        network (pandas dataframe): The return and how to interpret it

    Raises:
        IOError: An error occured accessing the database

    Tests:
        >>> get_nearest_node(-92.1647,37.7252)
        node_id = 634267, dist = 124
    """
    croppedRasterDF = rasterStatCroppedRaster(df,dem_path)
    appendedDF = gpd.GeoDataFrame(pd.concat([df,croppedRasterDF],axis=1))
    elevationChangeArrays = []
    totalRequiredHeightChanges = []
    totalCutFillVolumes = []
    for i,row in appendedDF.iterrows():
        maskedRaster = row['mini_raster_array']
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
    appendedDF['elevationChangeArray'] = elevationChangeArrays
    appendedDF['totalCutFillVolume'] = totalCutFillVolumes
    return appendedDF

## CURRENT TEST
# building the evaluation grid structure
"""
aoiDF = gpd.read_file("./test_data/geojson.json")
aoiDF.CRS = {'init':'epsg:4326'}
aoiDF = aoiDF.to_crs({'init':'epsg:3857'})
aoiPolygon = aoiDF.geometry[0]
aoiPolygon
import datetime
gridSpacing = 300
squareList = []
bounds = aoiPolygon.bounds
ll = bounds[:2]
ur = bounds[2:]
# https://stackoverflow.com/questions/30457089/how-to-create-a-polygon-given-its-point-vertices
start = datetime.datetime.now()
for x in floatrange(ll[0],ur[0],gridSpacing):
    for y in floatrange(ll[1],ur[1],gridSpacing):
        square = Polygon([[x,y],[x+gridSpacing,y],[x+gridSpacing,y+gridSpacing],[x,y+gridSpacing]])
        if square.within(aoiPolygon):
            squareList.append(square)
end = datetime.datetime.now()
end - start
timeElapsed = end - start
nFeatures = len(squareList)
print "Generated %s squares in %s seconds" %(nFeatures,timeElapsed.seconds)

evaluationGridDataFrame = gpd.GeoDataFrame(squareList)
evaluationGridDataFrame.columns = ['geometry']
plt.figure()
evaluationGridDataFrame.plot()
plt.savefig("./results/aoidiscretization.png")
df_cropped = evaluationGridDataFrame[0:100]
df_cropped.plot()

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
def minimumDistanceFromEvaluationToDataFrameFeatures(evaluationDF,vectorDF):
    minDistances = []
    for i,row in evaluationDF.iterrows():
        minDistance = vectorDF.distance(row.geometry).min()
        minDistances.append(minDistance)
    evaluationDF['distance'] = minDistances
    return evaluatioNDF

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

## TESTS
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


def testCreatingEmptyRaster():
    createEmptyRaster("./results/testemptyraster3.tif",lx,uy,30,100,100,3857)
    ds = gdal.Open('./results/testemptyraster3.tif')
    data = ds.ReadAsArray()
    plt.imshow(data)


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
