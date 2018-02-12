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
import shapely.geometry as geom
import rasterstats
from rasterstats import zonal_stats
import matplotlib.pyplot as plt
%matplotlib inline

""" REFERENCES
http://lxml.de/tutorial.html
https://mapbox.s3.amazonaws.com/playground/perrygeo/rasterio-docs/cookbook.html#rasterizing-geojson-features
http://skipperkongen.dk/2012/03/06/hello-world-of-raster-creation-with-gdal-and-python/
https://github.com/mapplus/qgis-scripts/blob/master/scripts/Raster%20Euclidean%20Distance%20Analysis.py
https://stackoverflow.com/questions/30740046/calculate-distance-to-nearest-feature-with-geopandas
http://portolan.leaffan.net/creating-sample-points-with-ogr-and-shapely-pt-2-regular-grid-sampling/
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


# CURRENT TEST
# extracting raster by extents to numpy array
r = gdal.Open(raster_path)
tmpPath = "./tmp/smallRaster.tif"
extents = aoiLoaded.bounds
extents
r = gdal.Translate(tmpPath,r,projWin=extents)

#https://gis.stackexchange.com/questions/159624/converting-large-rasters-to-numpy
import scipy
from scipy import misc
raster = misc.imread(raster_path)


# https://gis.stackexchange.com/questions/264793/crop-raster-in-memory-with-python-gdal-bindings
dataset = gdal.Open(raster_path)
band = dataset.GetRasterBand(1)
geotransform = dataset.GetGeoTransform()
xinit = geotransform[0]
yinit = geotransform[3]

xsize = geotransform[1]
ysize = geotransform[5]
#p1 = point upper left of bounding box
#p2 = point bottom right of bounding box
countyBounds = aoiLoaded.bounds
countyBounds
p1 = (-10287442.575418131, 4523429.485052726) #(6, 5)
p2 = (-10287342.575418131, 4523329.485052726) #(12, 14)
row1 = int((p1[1] - yinit)/ysize)
col1 = int((p1[0] - xinit)/xsize)

row2 = int((p2[1] - yinit)/ysize)
col2 = int((p2[0] - xinit)/xsize)

print "row1:%s,col1:%s,row2:%s,col2:%s" %(row1,col1,row2,col2)
data = band.ReadAsArray()
data.shape
# data = band.ReadAsArray(col1, row1, col2 - col1 + 1, row2 - row1 + 1)

#perform come calculations with ...
mean = np.mean(data)

# zonal stats
#https://pcjericks.github.io/py-gdalogr-cookbook/raster_layers.html
def zonal_stats(feat, input_zone_polygon, input_value_raster):

    # Open data
    raster = gdal.Open(input_value_raster)
    shp = ogr.Open(input_zone_polygon)
    lyr = shp.GetLayer()

    # Get raster georeference info
    transform = raster.GetGeoTransform()
    xOrigin = transform[0]
    yOrigin = transform[3]
    pixelWidth = transform[1]
    pixelHeight = transform[5]

    # Reproject vector geometry to same projection as raster
    sourceSR = lyr.GetSpatialRef()
    targetSR = osr.SpatialReference()
    targetSR.ImportFromWkt(raster.GetProjectionRef())
    coordTrans = osr.CoordinateTransformation(sourceSR,targetSR)
    feat = lyr.GetNextFeature()
    geom = feat.GetGeometryRef()
    geom.Transform(coordTrans)

    # Get extent of feat
    geom = feat.GetGeometryRef()
    if (geom.GetGeometryName() == 'MULTIPOLYGON'):
        count = 0
        pointsX = []; pointsY = []
        for polygon in geom:
            geomInner = geom.GetGeometryRef(count)
            ring = geomInner.GetGeometryRef(0)
            numpoints = ring.GetPointCount()
            for p in range(numpoints):
                    lon, lat, z = ring.GetPoint(p)
                    pointsX.append(lon)
                    pointsY.append(lat)
            count += 1
    elif (geom.GetGeometryName() == 'POLYGON'):
        ring = geom.GetGeometryRef(0)
        numpoints = ring.GetPointCount()
        pointsX = []; pointsY = []
        for p in range(numpoints):
                lon, lat, z = ring.GetPoint(p)
                pointsX.append(lon)
                pointsY.append(lat)

    else:
        sys.exit("ERROR: Geometry needs to be either Polygon or Multipolygon")

    xmin = min(pointsX)
    xmax = max(pointsX)
    ymin = min(pointsY)
    ymax = max(pointsY)

    # Specify offset and rows and columns to read
    xoff = int((xmin - xOrigin)/pixelWidth)
    yoff = int((yOrigin - ymax)/pixelWidth)
    xcount = int((xmax - xmin)/pixelWidth)+1
    ycount = int((ymax - ymin)/pixelWidth)+1

    # Create memory target raster
    target_ds = gdal.GetDriverByName('MEM').Create('', xcount, ycount, 1, gdal.GDT_Byte)
    target_ds.SetGeoTransform((
        xmin, pixelWidth, 0,
        ymax, 0, pixelHeight,
    ))

    # Create for target raster the same projection as for the value raster
    raster_srs = osr.SpatialReference()
    raster_srs.ImportFromWkt(raster.GetProjectionRef())
    target_ds.SetProjection(raster_srs.ExportToWkt())

    # Rasterize zone polygon to raster
    gdal.RasterizeLayer(target_ds, [1], lyr, burn_values=[1])

    # Read raster as arrays
    banddataraster = raster.GetRasterBand(1)
    dataraster = banddataraster.ReadAsArray(xoff, yoff, xcount, ycount).astype(numpy.float)

    bandmask = target_ds.GetRasterBand(1)
    datamask = bandmask.ReadAsArray(0, 0, xcount, ycount).astype(numpy.float)

    # Mask zone of raster
    zoneraster = numpy.ma.masked_array(dataraster,  numpy.logical_not(datamask))

    # Calculate statistics of zonal raster
    return numpy.average(zoneraster),numpy.mean(zoneraster),numpy.median(zoneraster),numpy.std(zoneraster),numpy.var(zoneraster)

# https://github.com/perrygeo/python-rasterstats
from rasterstats import zonal_stats
stats = zonal_stats('./test_data/MO_2016_TIGER_Counties_shp/MO_2016_TIGER_Counties_shp.shp',raster_path)

from rasterstats import point_query
point = "POINT(-10287442.575418131 4523429.485052726)"
point_query(point,raster_path)

# https://gis.stackexchange.com/questions/177035/geopandas-and-zonal-statistic-error
df['mean'] = gpd.GeoDataFrame(zonal_stats(vectors=df['geometry'],raster=raster_path,stats='mean'))['mean']
df.head()

# http://www.perrygeo.com/python-rasterstats.html
from rasterstats import raster_stats
veg_stats = raster_stats(df['geometry'], raster_path,
    stats="count majority minority unique",
    copy_properties=True,
    nodata_value=0,
    categorical=True)


# TESTS
# paths
xmlPath = "./input.xml"
raster_path = "/home/noah/FLW_Missouri Mission Folder/RASTER/DEM_CMB_ELV_SRTMVF2.tif"
vector_path = "./test_data/UtilityInfrastructureCrv_3.shp"

# FIRST PASS IMPLEMENTATION
df = gpd.read_file('./test_data/MO_2016_TIGER_Counties_shp/MO_2016_TIGER_Counties_shp.shp')
df.CRS = {'init':'epsg:4326'}
df_proj = df.to_crs({'init':'epsg:3857'})
aoiWKT = df_proj[df_proj["NAME"] == "Pulaski"].geometry.values[0].to_wkt()
aoiLoaded = loads(aoiWKT)

roadsDF = gpd.read_file(vector_path)
roadsDF.crs
roadsDF = roadsDF.to_crs({'init':'epsg:3857'})

lx,ly,ux,uy = df_proj['geometry'][4].envelope.bounds
filteredRoads = filterDataFrameByBounds(roadsDF,lx,ly,ux,uy)
filteredCounties = filterDataFrameByBounds(df_proj,lx,ly,ux,uy)
filteredCounties.plot(ax=filteredRoads.plot(facecolor='red'),facecolor='Green')



createEmptyRaster("./results/testemptyraster3.tif",lx,uy,30,100,100,3857)
ds = gdal.Open('./results/testemptyraster3.tif')
data = ds.ReadAsArray()
plt.imshow(data)



point = Point(ux,uy)
roadsDF.distance(point).min()


aoiLoaded = loads(aoiWKT)
testList = generateEvaluationGridDataFrame(aoiLoaded,3200)
testList.plot()
