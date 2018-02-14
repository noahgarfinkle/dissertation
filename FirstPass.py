# -*- coding: utf-8 -*-
"""Parses an XML file in order to evaluate sites
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

# IMPORTS
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

# HELPFUL FOR DEBUGGING
# %matplotlib inline
# pd.options.display.max_columns = 300

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

# CLASSES
class SiteSearch:
    def __init__(self):
        self.crs = None
        self.AreaOfInterest = None

class SiteSuitabilityCriteria:
    def __init__(self):
        return None

class SiteSuitabilityCriteria:
    def __init__(self):
        return None

class SiteRelationalConstraint:
    def __init__(self):
        return None

class Input:
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
        siteRelationalConstraints = self.root[1]
        siteRelationalConstraint = SiteRelationalConstraint()
        siteSearchRelationalConstraints = []
        siteRelationalConstraints.append(siteRelationalConstraint)
        return siteSearchRelationalConstraints

# SPATIAL FUNCTIONS
def distance():
    distances = gdal.distances()
    return None

def creatQAFRaster(crs,lx,ly,ux,uy,qafCellSize):
    numberXCells = np.ceiling((ux-lx)/qafCellSize)
    numberYCells = np.ceiling((uy-ly)/qafCellSize)

    newUX = lx + (numberXCells * qafCellSize)
    newUY = ly + (numberYCells * qafCellSize)

    qafMatrix = np.empty([numberXCells,numberYCells])
    driver = gdal.GetDriverByName("GTiff")
    dst_ds = driver.Create("./tmp/emtpyRaster.tif",numberXCells,numberYCells,0,gdal.GDT_Byte)
    pixel_width,pixel_height = qafCellSize
    topLeftX = lx
    topLeftY = newUY
    dst_ds.SetGeoTransform([topLeftX,pixel_width,0,topLeftY,0,-pixel_height])
    srs = osr.SpatialReference()
    srs.ImportFromEPSG(crs)
    dst_ds.SetProjection(srs.ExportToWkt())

def createQAFGridSurface():
    return 0

def buildEvaluationGridDataFrame(aoiJSON,exclusionJson,gridSpacing):
    df = gpd.GeoDataFrame()
    df.columns = ['geometry','total']

def filterDataFrameByBounds(df,lx,ly,ux,uy):
    filteredDF = df.cx[lx:ux,ly:uy]
    return filteredDF

def filterDataFrameByValue(df,column,argument):
    filteredDF = df[df[column]==argument]
    return filteredDF

def minimumDistanceFromPointToDataFrameFeatures(x,y,crs,df):
    point = Point(x,y)
    return df.distance(point).min()

def projectWKT(wkt,from_epsg,to_epsg):
    feat = loads(wkt)
    df_to_project = gpd.GeoDataFrame([feat])
    df_to_project.columns = ['geometry']
    df_to_project.crs = {'init':'EPSG:' + str(from_epsg)}
    df_to_project.to_crs({'init':'EPSG:' + str(to_epsg)})
    return df_to_project.geometry[0].to_wkt()

def buildSearchGrid(aoiWKT,aoiWKTProjection=4326,gridSpacing=30,exclusionFeatures = []):
    numberXCells = np.ceiling((ux-lx)/qafCellSize)
    numberYCells = np.ceiling((uy-ly)/qafCellSize)

    newUX = lx + (numberXCells * qafCellSize)
    newUY = ly + (numberYCells * qafCellSize)

    qafMatrix = np.empty([numberXCells,numberYCells])

def createEmptyRaster(rasterPath,topLeftX,topLeftY,cellSize,width,height,epsg):
    geotransform = [topLeftX,cellSize,0,topLeftY,0,-cellSize]
    driver = gdal.GetDriverByName("GTiff")
    dst_ds = driver.Create(rasterPath, width, height, 1, gdal.GDT_Byte )
    dst_ds.SetGeoTransform(geotransform)
    srs = osr.SpatialReference()
    srs.ImportFromEPSG(epsg)
    dst_ds.SetProjection(srs.ExportToWkt())
    raster = np.zeros((height,width),dtype=np.uint32)
    raster[25:50,25:50] = 100 # this code is for testing
    dst_ds.GetRasterBand(1).WriteArray(raster)


def floatrange(start, stop, step):
    while start < stop:
        yield start
        start += step

def generateEvaluationGridDataFrame(polygon,gridSpacing):
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
    evaluationGridDataFrame.columns = ['geometry']
    return evaluationGridDataFrame

def convertRasterToNumpyArray(raster_path):
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

validStats = utils.VALID_STATS
def generateRasterStatisticsForDataFrame(df,raster_Path,stats="count majority minority unique mean",isCategorical=False):
    row_stats_df = gpd.GeoDataFrame(raster_stats(vectors=df['geometry'],raster=raster_path,stats=stats, copy_properties=True, nodata_value=0, categorical=isCategorical))
    newDF = gpd.GeoDataFrame(pd.concat([df,row_stats_df],axis=1))
    return newDF

# Note: Requires point in same coordinate system as raster, I assume
# todo: implement projections
def queryRasterValueForPoint(x,y,raster_path,pointCRS=None,rasterCRS=None):
    point = "POINT(%s %s)" %(x,y)
    return point_query(point,raster_path)

def rasterStatCroppedRaster(df,raster_path):
    rasterSource = zonal_stats(df['geometry'],raster_path,all_touched=True,raster_out=True)
    rasterDF = pd.DataFrame(rasterSource)
    return rasterDF

#finalElevation must either be number or string from validStats
def calculateCutFill(df,dem_path,finalElevation='mean',rasterResolution=30):
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
            print row[finalElevation]
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

# CURRENT TEST
import os
os.getcwd()

# TESTS
# paths
xmlPath = "./input.xml"
raster_path = "../FLW_Missouri Mission Folder/RASTER/DEM_CMB_ELV_SRTMVF2.tif"
vector_path = "./test_data/UtilityInfrastructureCrv_3.shp"

# FIRST PASS IMPLEMENTATION
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
