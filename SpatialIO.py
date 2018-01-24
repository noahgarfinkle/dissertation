# -*- coding: utf-8 -*-
"""Handles spatial data management for my dissertation

This module creates wrapper classes for the spatial data
structures required for my dissertation, equipping each
with my most commonly used operators in order to replace
R and it's great spatial support from my dissertation.
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
__date_created__ = "20 January 2018"

## Imports
import doctest
import folium
import folium.plugins as plugins
import numpy as np
import geopandas as gpd
from datetime import datetime, timedelta
from shapely.geometry import Point
import psycopg2
from shapely.geometry import Point, Polygon
from enum import Enum
import matplotlib.pyplot as plt
from osgeo import gdal, ogr, osr
from scipy.ndimage import imread
from os.path import exists
from osgeo.gdalconst import GA_ReadOnly
from struct import unpack
from os.path import exists
from numpy import logical_and
from numpy import zeros
from numpy import uint8
import scipy
import os
from PIL import Image, ImageChops

%matplotlib inline

## Enumerations
class CRS(Enum):
    WGS84 = 4326
    WMAS = 3857

## Data Structures
class PostGIS:
    def __init__(self,dbname,user='postgres',host='localhost',password='postgres'):
        connString = "dbname='%s' user='%s' host='%s' password='%s'" %(dbname,user,host,password)
        try:
            self.conn = psycopg2.connect(connString)
        except:
            self.conn = None
            print "Unable to connect to database"

    def query(self,sql):
        cur = self.conn.cursor()
        cur.execute(sql)
        rows = cur.fetchall()
        return rows

    def queryToDF(self,sql,geom_col='geom'):
        try:
            df = gpd.read_postgis(sql,con=self.conn,geom_col=geom_col)
            return df
        except:
            print "Unable to execute query"


class GeoDataFrame:
    def __init__(self):
        self.df = None
        self.name = "Not Set"
        self.type = "Point"
        self.crs = None

    def createGeoDataFrame(self,crs,columns=['geometry']):
        crsDict = {'init':'epsg:%s' %(crs.value)}
        self.df = gpd.GeoDataFrame(crs=crsDict,columns=columns)
        self.crs = crs

    def from_shapefile(self,filePath):
        return None

    def from_json(self,json):
        return None

    def from_postgis(self,postgis,query):
        return None

    def addColumn(self,colName):
        self.df[colName] = None

    def addRow(self,mapping):
        self.df = self.df.append(mapping,ignore_index=True)

    def plot(self,column=None):
        self.df.plot(column=column)

    def to_shapefile(self,filePath):
        self.df.to_file(filePath,driver="ESRI Shapefile")

    def to_json(self):
        json = self.df.to_json()
        return json

    def reproject(self,toCRS):
        self.df.to_crs(epsg=toCRS.value,inplace=True)
        self.crs = toCRS

    def summary(self):
        print self.df.head()


class RasterLayer:
    def __init__(self,name="Not set"):
        self.name = name
        self.rasterPath = None
        self.raster = None
        self.crs = None
        self.scale = None
        self.minimum = None
        self.maximum = None
        self.unitType = None
        self.colorInterpretation = None
        self.colorTable = None
        self.lx = None
        self.ly = None
        self.ux = None
        self.uy = None


    def from_empty(self,lx,ly,ux,uy,crs,scale):
        return 0

    def from_file(self,raster_path):
        self.rasterPath = raster_path
        self.raster = gdal.Open(raster_path)
        proj = self.raster.GetProjection()
        srs = osr.SpatialReference(wkt=proj)
        self.crs = srs.ExportToWkt()
        self.scale = self.raster.GetRasterBand(1).GetScale()
        self.minimum = self.raster.GetRasterBand(1).GetMinimum()
        self.maximum = self.raster.GetRasterBand(1).GetMaximum()
        self.unitType = self.raster.GetRasterBand(1).GetUnitType()
        self.colorInterpretation = self.raster.GetRasterBand(1).GetColorInterpretation()
        self.colorTable = self.raster.GetRasterBand(1).GetColorTable()
        # get the extents, https://gis.stackexchange.com/questions/104362/how-to-get-extent-out-of-geotiff
        geoTransform = self.raster.GetGeoTransform()
        self.lx = geoTransform[0]
        self.uy = geoTransform[3]
        self.ux = self.lx + geoTransform[1] * self.raster.RasterXSize
        self.ly = self.uy + geoTransform[5] * self.raster.RasterYSize

    def plot(self):
        return 0

    def export(self,newPath):
        self.rasterPath = newPath

    def reproject(self,crs=CRS.WMAS):
        tmpRaster = "./tmp/tmp.tif"
        spatRef = osr.SpatialReference()
        spatRef.ImportFromEPSG(crs.value)
        gdal.Warp(tmpRaster,self.raster,dstSRS=spatRef)
        self.raster = gdal.Open(tmpRaster)
        self.crs = spatRef.ExportToWkt()
        self.rasterPath = "In memory: export to update"

    def crop(self,lx,ly,ux,uy):
        result = RasterLayer()
        return result

    def crop(self,lx,ly,ux,uy):
        return 0

    def toPNG(self,outputPath):
        argument = "gdaldem hillshade -of PNG %s %s" %(self.rasterPath,outputPath)
        cdArgument = "cd /home/noah/GIT/dissertation/results"
        os.system(cdArgument)
        os.system(argument)
        return argument



class VectorLayer:
    def __init__(self,name="Not set"):
        self.df = None
        self.name = name

    def loadFeatureLayerFromFile(self,filePath):
        #Support shapefile, geojson
        self.df = gpd.read_file(filePath)

    def crop(self,lx,ly,ux,uy):
        return 0

    def loadFeatureLayerFromPostGIS(self,con,sql,geom_col='geom',crs=None,index_col=None,coerce_float=True,params=None):
        self.df = gpd.read_postgis(sql,con,geom_col=geom_col,crs=crs,index_col=index_col,params=params)


class Map:
    def __init__(self,name="Not set"):
        self.map = folium.Map([48., 5.], tiles='stamentoner', zoom_start=6)
        self.name = name

    def addRasterLayerAsOverlay(self,rasterLayer,opacity):
        # http://qingkaikong.blogspot.in/2016/06/using-folium-5-image-overlay-overlay.html
        # http://nbviewer.jupyter.org/github/python-visualization/folium/blob/master/examples/ImageOverlay.ipynb
        # http://nbviewer.jupyter.org/github/ocefpaf/folium_notebooks/blob/master/test_image_overlay_gulf_stream.ipynb
        # 1. get boundary of raster
        bounds =[[rasterLayer.ly,rasterLayer.lx], [rasterLayer.uy,rasterLayer.ux]]

        # 2. export raster to png
        data = np.array(rasterLayer.raster.GetRasterBand(1).ReadAsArray())
        pngPath = "./tmp/temppng.png"
        rasterLayer.toPNG(pngPath)
        img = Image.open(pngPath)

        # 3. add ImageOverlay
        self.map.add_children(plugins.ImageOverlay(data,opacity=opacity,bounds=bounds))

    def addVectorLayerAsOverlay(self,vectorLayer):
        # https://ocefpaf.github.io/python4oceanographers/blog/2015/12/14/geopandas_folium/
        # http://andrewgaidus.com/leaflet_webmaps_python/
        return 0

    def saveMap(self,filePath):
        self.map.save(filePath)
        print("Map saved to %s" %(filePath))


# test composite

map = Map(name="test map")
rl = RasterLayer(name="test raster")
rl.from_file("/home/noah/GIT/dissertation/test_data/testelevunproj.tif")
vl = VectorLayer(name="test vector")
rl.toPNG("./tmp/testout5.png")
rl.lx
rl.ux
rl.ly
rl.uy
map.addRasterLayerAsOverlay(rl,0.5)
map.saveMap("./results/addingrasterlayer.html")




"""
Manages all test functions for SpatialIO
"""
def test():
    doctest.testmod()


#def testGeoDataFrame():
g = GeoDataFrame()
g.createGeoDataFrame(CRS.WMAS,columns=['geometry','a'])
g.addColumn('b')
g.addRow({'geometry':Point(49,50),'a':1,'b':'c'})
print g.crs
g.plot()
print g.to_json()
g.to_shapefile("./results/test.shp")
g.reproject(CRS.WGS84)
g.plot()
print g.crs
