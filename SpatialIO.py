# -*- coding: utf-8 -*-
"""Handles spatial data management for my dissertation

This module creates wrapper classes for the spatial data
structures required for my dissertation, equipping each
with my most commonly used operators in order to replace
R and it's great spatial support from my dissertation.

Todo:
    * Connect with the other modules
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

## HELPFUL FOR DEBUGGING
# %matplotlib inline
# pd.options.display.max_columns = 300

## SETUP

""" REFERENCES

"""

## Enumerations
class CRS(Enum):
    """ Relates CRS to EPSG values to streamline projections
    """
    WGS84 = 4326
    WMAS = 3857

## CLASSES
class PostGIS:
    """ Encapsulates onnections to PostGIS database

        Will be used for connecting to ENSITE database, routing table, and allowing
        access to other databases

        Attributes:
            dbname (str): Name of the database
            user (str): Postgres user name
            host (str): Host path to the database, defaults localhost
            password (str): Postgres password
    """

    def __init__(self,dbname,user='postgres',host='localhost',password='postgres'):
        connString = "dbname='%s' user='%s' host='%s' password='%s'" %(dbname,user,host,password)
        try:
            self.conn = psycopg2.connect(connString)
        except:
            self.conn = None
            print "Unable to connect to database"

    def query(self,sql):
        """ Submits a SQL query to the postgres connection and returns the result


        Args:
            sql (str): A string containing the properly formated SQL expression for
                Postgres

        Returns:
            rows (cursor results): This is not the prefered query method, see queryToDF

        Raises:
            None

        Tests:
        """
        cur = self.conn.cursor()
        cur.execute(sql)
        rows = cur.fetchall()
        return rows

    def queryToDF(self,sql,geom_col='geom'):
        """ Submits a SQL query to the postgres connection and returns a geopandas dataframe

        Assumes that the query is seeking spatial information

        Args:
            sql (str): A string containing the properly formated SQL expression for
                Postgres
            geom_col (str): The name of the column in Postgis which contains a valid
                geometry, so that it can be translated into a GeoPandas geometry

        Returns:
            df (GeoPandas dataframe): Geometry is collected from geom_col and contained
                in 'geometry'

        Raises:
            None

        Tests:
            None
        """
        try:
            df = gpd.read_postgis(sql,con=self.conn,geom_col=geom_col)
            return df
        except:
            print "Unable to execute query"


class GeoDataFrame:
    """ Wrapper for GeoPandas GeoDataFrame

        Designed to make it more convenient to interact with certain workflows in
        Geopandas

        Attributes:
            df (GeoPandas GeoDataFrame):
            name (str): The pretty-print name of the data being stored, for convenience
            type (str): The datastructure being stored, for convenience
            crs (ENUM CRS): The projection being stored, for convenience
    """

    def __init__(self):
        self.df = None
        self.name = "Not Set"
        self.type = "Point"
        self.crs = None

    def createGeoDataFrame(self,crs,columns=['geometry']):
        """Creates an empty GeoDataFrame

        Args:
            crs (ENUM CRS): The first parameter.
            columns (list:str): The columns this database should include.  Should
            include a 'geometry' column for spatial data

        Returns:
            None: Sets the classes dataframe and crs

        Raises:
            None

        Tests:
            None
        """
        crsDict = {'init':'epsg:%s' %(crs.value)}
        self.df = gpd.GeoDataFrame(crs=crsDict,columns=columns)
        self.crs = crs

    def from_shapefile(self,filePath):
        """ Creates the GeoDataFrame from the path to a shapefile

        Args:
            filePath (str): The full path, without backslashes, to the .shp portion
                of a shapefile

        Returns:
            None: Sets the classes dataframe and crs

        Raises:
            None

        Tests:
            None
        """
        return None

    def from_json_file(self,json):
        """ Creates the GeoDataFrame from the path to a json file

        Args:
            filePath (str): The full path, without backslashes, to a GeoJSON file

        Returns:
            None: Sets the classes dataframe and crs

        Raises:
            None

        Tests:
            None
        """
        return None

    def from_json_str(self,json):
        """ Creates the GeoDataFrame from a GeoJSON string

        Args:
            filePath (str): A properly formatted GeoJSON string

        Returns:
            None: Sets the classes dataframe and crs

        Raises:
            None

        Tests:
            None
        """
        return None

    def from_postgis(self,postgis,query):
        """ Creates the GeoDataFrame from a PostGIS query

        Args:
            postgis (PostGIS): A object of type PostGIS, already properly configured
            query (str): A sql string to be passed to the postgis object queryToDF
                function

        Returns:
            None: Sets the classes dataframe and crs

        Raises:
            None

        Tests:
            None
        """
        return None

    def addColumn(self,colName):
        """ Adds an empty column to the dataframe

        Requires that a dataframe has already been instantiated

        Args:
            colName (str): The name of the column in the revised dataframe

        Returns:
            None

        Raises:
            None

        Tests:
            None
        """
        self.df[colName] = None

    def addRow(self,mapping):
        """ Insert a row into the geodataframe

        Not well implemented yet

        Args:
            mapping (var): The row to insert

        Returns:
            None

        Raises:
            None

        Tests:
            None
        """
        self.df = self.df.append(mapping,ignore_index=True)

    def plot(self,column=None):
        """ Utilizes the GeoDataFrame default plot behavior

        Not well implemented yet

        Args:
            column (str): If a valid column is passed, produces a choropleth plot
                using that column

        Returns:
            None

        Raises:
            None

        Todo:
            * Option to save the plot to a file

        Tests:
            None
        """
        self.df.plot(column=column)

    def to_shapefile(self,filePath):
        """ Writes the GeoDataFrame to an ESRI shapefile

        Args:
            filePath (str): The filepath to write the shapefile files to

        Returns:
            None

        Raises:
            None

        Tests:
            None
        """
        self.df.to_file(filePath,driver="ESRI Shapefile")

    def to_json(self):
        """ Writes the GeoDataFrame to a GeoJSON file

        Args:
            filePath (str): The filepath to write the GeoJSON file to

        Returns:
            None

        Raises:
            None

        Tests:
            None
        """
        json = self.df.to_json()
        return json

    def reproject(self,toCRS):
        """ Reprojects the GeoDataFrame, replacing it

        Args:
            toCRS (ENUM CRS): The target projection for the data frame

        Returns:
            None

        Raises:
            None

        Tests:
            None
        """
        self.df.to_crs(epsg=toCRS.value,inplace=True)
        self.crs = toCRS

    def summary(self):
        """ Prints the GeoDataFrame 'head()'' function

        Args:
            None

        Returns:
            None

        Raises:
            None

        Tests:
            None
        """
        print self.df.head()


class RasterLayer:
    """ Summary of class

        Longer class information

        Attributes:
            attr1 (str): The first attribute
            attr2 (int): The second attribute
    """

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
        return 0

    def from_file(self,raster_path):
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
        self.rasterPath = newPath

    def reproject(self,crs=CRS.WMAS):
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
        tmpRaster = "./tmp/tmp.tif"
        spatRef = osr.SpatialReference()
        spatRef.ImportFromEPSG(crs.value)
        gdal.Warp(tmpRaster,self.raster,dstSRS=spatRef)
        self.raster = gdal.Open(tmpRaster)
        self.crs = spatRef.ExportToWkt()
        self.rasterPath = "In memory: export to update"

    def crop(self,lx,ly,ux,uy):
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
        result = RasterLayer()
        return result

    def toPNG(self,outputPath):
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
        argument = "gdaldem hillshade -of PNG %s %s" %(self.rasterPath,outputPath)
        cdArgument = "cd /home/noah/GIT/dissertation/results"
        os.system(cdArgument)
        os.system(argument)
        return argument



class VectorLayer:
    """ Summary of class

        Longer class information

        Attributes:
            attr1 (str): The first attribute
            attr2 (int): The second attribute
    """

    def __init__(self,name="Not set"):
        self.df = None
        self.name = name

    def loadFeatureLayerFromFile(self,filePath):
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
        #Support shapefile, geojson
        self.df = gpd.read_file(filePath)

    def crop(self,lx,ly,ux,uy):
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
        return 0

    def loadFeatureLayerFromPostGIS(self,con,sql,geom_col='geom',crs=None,index_col=None,coerce_float=True,params=None):
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
        self.df = gpd.read_postgis(sql,con,geom_col=geom_col,crs=crs,index_col=index_col,params=params)


class Map:
    """ Summary of class

        Longer class information

        Attributes:
            attr1 (str): The first attribute
            attr2 (int): The second attribute
    """

    def __init__(self,name="Not set"):
        self.map = folium.Map([48., 5.], tiles='stamentoner', zoom_start=6)
        self.name = name

    def addRasterLayerAsOverlay(self,rasterLayer,opacity):
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
        # https://ocefpaf.github.io/python4oceanographers/blog/2015/12/14/geopandas_folium/
        # http://andrewgaidus.com/leaflet_webmaps_python/
        gjson = vectorLayer.df.to_crs('3857').to_json()
        features = folium.features.GeoJson(gjson)
        self.map.add_children(features)


    def saveMap(self,filePath):
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
        self.map.save(filePath)
        print("Map saved to %s" %(filePath))

    def addCoolIcon(self,lat,lon,icon):
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
        # https://www.kaggle.com/daveianhickey/how-to-folium-for-maps-heatmaps-time-series
        coolIcon = folium.Marker([51.5183, 0.3206],
              popup='East London',
              icon=folium.Icon(color='blue',icon='bar-chart', prefix='fa')
             )
             # icon=folium.Icon(color='red',icon='bicycle', prefix='fa')
        self.map.add_child(coolIcon)

    def generateRandomLatLonPair(self,latMin,latMax,lonMin,lonMax):
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
        lat = np.random.uniform(latMin,latMax)
        lon = np.random.uniform(lonMin,lonMax)
        return lat,lon

    def addTimeSeriesHeatMap(self):
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
        heat_data = []
        for gen in range(0,5):
            gen = []
            for i in range(0,1001):
                lat,lon = self.generateRandomLatLonPair(35,50,-90,-80)
                #lat = np.random.randint(35,50)
                #lon = np.random.randint(-90,-80)
                val = [lat,lon]
                gen.append(val)
            heat_data.append(gen)
        hm = plugins.HeatMapWithTime(heat_data)
        self.map.add_child(hm)


class GenerationalSolutions:
    """ Summary of class

        Longer class information

        Attributes:
            attr1 (str): The first attribute
            attr2 (int): The second attribute
    """
    def __init__(self):
        self.dataFrame = GeoDataFrame()
        self.dataFrame.createGeoDataFrame(CRS.WMAS,columns=['geometry','generation','score'])

    def addPoint(self,lat,lon,generation,score):
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
        geom = Point(lon,lat)
        self.dataFrame.addRow({'geometry':geom,'generation':generation,'score':score})


## CURRENT TEST

# Mapping with tables
# https://ocefpaf.github.io/python4oceanographers/blog/2015/12/14/geopandas_folium/
import folium

mapa = folium.Map([-15.783333, -47.866667],
                  zoom_start=4,
                  tiles='cartodbpositron')

points = folium.features.GeoJson(gjson)

mapa.add_children(points)
mapa


table = """
<!DOCTYPE html>
<html>
<head>
<style>
table {{
    width:100%;
}}
table, th, td {{
    border: 1px solid black;
    border-collapse: collapse;
}}
th, td {{
    padding: 5px;
    text-align: left;
}}
table#t01 tr:nth-child(odd) {{
    background-color: #eee;
}}
table#t01 tr:nth-child(even) {{
   background-color:#fff;
}}
</style>
</head>
<body>

<table id="t01">
  <tr>
    <td>Type</td>
    <td>{}</td>
  </tr>
  <tr>
    <td>Name</td>
    <td>{}</td>
  </tr>
  <tr>
    <td>Operational</td>
    <td>{}</td>
  </tr>
</table>
</body>
</html>
""".format

## TESTS
# test composite
def testComposite():
    map = Map(name="test map")
    map.addTimeSeriesHeatMap()
    rl = RasterLayer(name="test raster")
    #rl.from_file("/home/noah/GIT/dissertation/test_data/testelevunproj.tif")
    rl.from_file("./test_data/testelevunproj.tif")
    vl = VectorLayer(name="test vector")
    rl.toPNG("./tmp/testout5.png")
    rl.lx
    rl.ux
    rl.ly
    rl.uy
    map.addRasterLayerAsOverlay(rl,0.5)
    map.addCoolIcon(38.878057,-90.28944,'bar-chart')
    map.saveMap("./results/testHeatMapWithTime_worklaptop.html")

    g = GenerationalSolutions()
    g.addPoint(50,-93,1,1)
    g.dataFrame.df
    g.dataFrame.df.plot()

"""
Manages all test functions for SpatialIO
"""
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
