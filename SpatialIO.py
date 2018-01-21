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
class CRS(Enum):
    WGS84 = 4326
    WMAS = 3857

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

    def createGeoDataFrame(self,crs,columns=['geometry']):
        crsDict = {'init':'epsg:%s' %(crs.value)}
        self.df = gpd.GeoDataFrame(crs=crsDict,columns=columns)

    def addColumn(self,colName):
        self.df[colName] = None

    def addRow(self,mapping):
        self.df = self.df.append(mapping,ignore_index=True)

    def plot(self):
        self.df.plot()

g = GeoDataFrame()
g.createGeoDataFrame(CRS.WMAS,columns=['geometry','a','b'])
g.addRow({'geometry':Point(49,50),'a':1,'b':'c'})
g.plot()

class RasterLayer:
    def __init__(self):
        return None

    def crop(self,lx,ly,ux,uy):
        result = RasterLayer()
        return result


class VectorLayer:
    def __init_(self):
        self.df = None

    def loadFeatureLayerFromFile(self,filePath):
        """
        Support shapefile, geojson
        """
        self.df = gpd.read_file(filePath)

    def loadFeatureLayerFromPostGIS(self,con,sql,geom_col='geom',crs=None,index_col=None,coerce_float=True,params=None):
        """

        """
        self.df = gpd.read_postgis(sql,con,geom_col=geom_col,crs=crs,index_col=index_col,params=params)


class PointLayer:
    def __init_(self):
        return None


class LineLayer:
    def __init_(self):
        return None


class PolygonLayer:
    def __init_(self):
        return None


class Map:
    def __init__(self):
        self.map = folium.Map([48., 5.], tiles='stamentoner', zoom_start=6)

    def addRasterLayerAsOverlay(self,rasterLayer):
        # http://qingkaikong.blogspot.in/2016/06/using-folium-5-image-overlay-overlay.html
        # http://nbviewer.jupyter.org/github/python-visualization/folium/blob/master/examples/ImageOverlay.ipynb
        return 0

    def addVectorLayerAsOverlay(self,vectorLayer):
        # https://ocefpaf.github.io/python4oceanographers/blog/2015/12/14/geopandas_folium/
        # http://andrewgaidus.com/leaflet_webmaps_python/
        return 0

    def saveMap(self,filePath):
        self.map.save(filePath)
        print("Map saved to %s" %(filePath))

"""
Manages all test functions for SpatialIO
"""output_file("burtin.html", title="burtin.py example")

def test():
    doctest.testmod()
