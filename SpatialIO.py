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

    def from_empty(self,lx,ly,ux,uy,crs,scale):
        return 0

    def from_file(self,raster_path):
        self.rasterPath = raster_path
        proj = self.raster.GetProjection()
        srs = osr.SpatialReference(wkt=proj)
        #epsg = srs.GetAttrValue('AUTHORITY',1) # https://gis.stackexchange.com/questions/267321/extracting-epsg-from-a-raster-using-gdal-bindings-in-python
        self.crs = srs.ExportToWkt()

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

class VectorLayer:
    def __init_(self,name="Not set"):
        self.df = None
        self.name = name

    def loadFeatureLayerFromFile(self,filePath):
        """
        Support shapefile, geojson
        """
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
        # 1. get boundary of raster
        min_lon, max_lon, min_lat, max_lat = rasterLayer.raster.GetExtent()
        bounds =[[min_lat, min_lon], [max_lat, max_lon]]

        # 2. export raster to png
        data = np.array(rasterLayer.raster.GetRasterBand(1).ReadAsArray())

        # 3. add ImageOverlay
        self.map.add_children(plugins.ImageOverlay(data,opacity=opacity,bounds=bounds))

    def addVectorLayerAsOverlay(self,vectorLayer):
        # https://ocefpaf.github.io/python4oceanographers/blog/2015/12/14/geopandas_folium/
        # http://andrewgaidus.com/leaflet_webmaps_python/
        return 0

    def saveMap(self,filePath):
        self.map.save(filePath)
        print("Map saved to %s" %(filePath))


def testComposit():
    map = Map(name="test map")
    rl = RasterLayer(name="test raster")
    


"""
Manages all test functions for SpatialIO
"""output_file("burtin.html", title="burtin.py example")
def test():
    doctest.testmod()


def testGeoDataFrame():
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
