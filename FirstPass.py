# -*- coding: utf-8 -*-
"""Parses an XML file in order to evaluate sites
"""

__author__ = "Noah W. Garfinkle"
__copyright__ = "Copyright 2018, Noah W. Garfinkle"
__credits__ = ["Dr. Ximing Cai", "Dr. George Calfas", "Thomas 'Max' Foltz",
                    "Juliana McMillan-Wilhoit", "Matthew Hiett",
                    "Dylan Pasley"]
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
import geopandas as gpd
import fiona
import pyproj
import rasterio
import numpy as np
import math
import shapely
from shapely.geometry import Point, Polygon
from shapely.wkt import loads
import matplotlib.pyplot as plt
%matplotlib inline

""" REFERENCES
http://lxml.de/tutorial.html
https://mapbox.s3.amazonaws.com/playground/perrygeo/rasterio-docs/cookbook.html#rasterizing-geojson-features
http://skipperkongen.dk/2012/03/06/hello-world-of-raster-creation-with-gdal-and-python/
"""

# CLASSES
class SiteSearch:
    def __init__(self):
        self.crs = None
        self.AreaOfInterest = None

class SiteSuitabilityCriteria:
    def __init__(self):
        return None

class SiteSuitabilityCriteriaEvaluahttps://mapbox.s3.amazonaws.com/playground/perrygeo/rasterio-docs/cookbook.html#rasterizing-geojson-featurestion:
    def __init__(self):EvalFallBack
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
qafCellSize
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

# TESTS
xmlPath = "./input.xml"

# test distance
raster_path = "/home/noah/FLW_Missouri Mission Folder/RASTER/DEM_CMB_ELV_SRTMVF2.tif"
raster = gdal.Open(raster_path)
raster_array = raster.ReadAsArray()
%matplotlib inline
plt.imshow(raster_array)
vector_path = "/home/noah/FLW_Missouri Mission Folder/VECTOR/TransportationGroundCrv.shp"
driver = ogr.GetDriverByName('ESRI Shapefile')
lines = driver.Open(vector_path,0)
linesLayer = lines.GetLayer()

# rasterize the vector
# https://github.com/mapplus/qgis-scripts/blob/master/scripts/Raster%20Euclidean%20Distance%20Analysis.py


rasterizedLinesLayer = gdal.RasterizeLayer(linesLayer)


lat_min="-4.50181532782252"
lon_min="39.315948486328125"
lat_max="-4.2738327000745"
lon_max="39.5452880859375"
qafCellSize="30"

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



# shapely distances
import fiona
import shapely.geometry
vct = fiona.open(vector_path)


# https://stackoverflow.com/questions/30740046/calculate-distance-to-nearest-feature-with-geopandas
%matplotlib inline
import matplotlib.pyplot as plt
import shapely.geometry as geom
import numpy as np
import pandas as pd
import geopandas as gpd

lines = gpd.GeoSeries(
    [geom.LineString(((1.4, 3), (0, 0))),
        geom.LineString(((1.1, 2.), (0.1, 0.4))),
        geom.LineString(((-0.1, 3.), (1, 2.)))])

# 10 points
n  = 10
points = gpd.GeoSeries([geom.Point(x, y) for x, y in np.random.uniform(0, 3, (n, 2))])

# Put the points in a dataframe, with some other random column
df_points = gpd.GeoDataFrame(np.array([points, np.random.randn(n)]).T)
df_points.columns = ['geometry', 'Property1']

points.plot()
lines.plot()
min_dist = np.empty(n)
for i, point in enumerate(points):
    min_dist[i] = np.min([point.distance(line) for line in lines])
df_points['min_dist_to_lines'] = min_dist
df_points.head(3)

def min_distance(point, lines):
    return lines.distance(point).min()
df_lines = gpd.GeoDataFrame(lines)
df_lines.columns = ['geometry']
min_distance(df_points.geometry[0],df_lines)

df_points['min_dist_to_lines'] = df_points.geometry.apply(min_distance, df_lines)
df_points.head()

road_df = gpd.read_file(vector_path)


# FIRST PASS IMPLEMENTATION
df = gpd.read_file('./test_data/MO_2016_TIGER_Counties_shp/MO_2016_TIGER_Counties_shp.shp')
df.head()
df.plot(column="NAME")
aoiJSON = df['geometry'][0].to_wkt()
aoiJSON
df.CRS = {'init':'epsg:4326'}
df_proj = df.to_crs({'init':'epsg:3857'})feat.project({"init":"EPSG:4326"})
df_proj.head()
df_proj.plot(column="NAME")

aoiWKT = df_proj['geometry'][0].to_wkt()
aoiLoaded = loads(aoiWKT)

def filterDataFrameByBounds(df,lx,ly,ux,uy):
    filteredDF = df.cx[lx:ux,ly:uy]
    return filteredDF
df_proj
def filterDataFrameByValue(df,column,argument):
    filteredDF = df[df[column]==argument]
    return filteredDF


vector_path = "./test_data/UtilityInfrastructureCrv_3.shp"
roadsDF = gpd.read_file(vector_path)
roadsDF.crs
roadsDF.plot()
roadsDF = roadsDF.to_crs({'init':'epsg:3857'})

lx,ly,ux,uy = df_proj['geometry'][4].envelope.bounds
filteredRoads = filterDataFrameByBounds(roadsDF,lx,ly,ux,uy)
filteredRoads.plot()
filteredCounties = filterDataFrameByBounds(df_proj,lx,ly,ux,uy)
filteredCounties.plot(ax=filteredRoads.plot(facecolor='red'),facecolor='Green')



"""
"""
def buildSearchGrid(aoiWKT,aoiWKTProjection=4326,gridSpacing=30,exclusionFeatures = []):
    return 0



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
createEmptyRaster("./results/testemptyraster3.tif",lx,uy,30,100,100,3857)
ds = gdal.Open('./results/testemptyraster3.tif')
data = ds.ReadAsArray()
data = np.flipud(data)
plt.imshow(data)

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

point = Point(ux,uy)
lx
uy
roadsDF.distance(point).min()
roadsDF.plot()
