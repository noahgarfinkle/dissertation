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
from shapely.geometry import Point, Polygon, LineString, MultiLineString
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
from PIL import Image, ImageChops
import rasterio
import fiona
from rasterio import features
import os
import pandas as pd
import sys

## HELPFUL FOR DEBUGGING
# %matplotlib inline
# pd.options.display.max_columns = 300

## LIFESAVING RESOURCES
# http://www.thisprogrammingthing.com/2013/fixing-the-this-exceeds-githubs-file-size-limit-of-100-mb-error/

## SETUP

""" REFERENCES
http://qingkaikong.blogspot.in/2016/06/using-folium-5-image-overlay-overlay.html
http://nbviewer.jupyter.org/github/python-visualization/folium/blob/master/examples/ImageOverlay.ipynb
http://nbviewer.jupyter.org/github/ocefpaf/folium_notebooks/blob/master/test_image_overlay_gulf_stream.ipynb
https://ocefpaf.github.io/python4oceanographers/blog/2015/12/14/geopandas_folium/
http://andrewgaidus.com/leaflet_webmaps_python/
https://www.kaggle.com/daveianhickey/how-to-folium-for-maps-heatmaps-time-series
https://gist.github.com/mhweber/1a07b0881c2ab88d062e3b32060e5486
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


class VectorLayer:
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
        self.df = gpd.read_file(filePath)

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

    def from_postgis(self,postgis,query,geom_col='geom',crs=None,index_col=None,coerce_float=True,params=None):
        """ Creates the GeoDataFrame from a PostGIS query

        Args:
            postgis (PostGIS): A object of type PostGIS, already properly configured
            query (str): A sql string to be passed to the postgis object queryToDF
                function
                geom_col (str): The column of the returned query to be treated
                    as a PostGIS geometry column
                crs (ENUM CRS): The projection to store the data in
                index_col (str): The column of the returned query to be treated
                    as the index
                corece_float (bool): Converts numbers to float if possible if True.
                    Otherwise, if False, considers all values to be strings.
                params (var): Not sure, I think this might be a dictionary

        Returns:
            None: Sets the classes dataframe

        Raises:
            None

        Tests:
            None
        """
        self.df = gpd.read_postgis(sql,postgis.con,geom_col=geom_col,crs=crs,index_col=index_col,params=params)

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

    def crop(self,lx,ly,ux,uy):
        """ Crops the dataframe to a bounding box

        Args:
            lx (float): the x-coordiante of the lower-left
            ly (float): the y-coordiante of the lower-left
            ux (float): the x-coordiante of the upper-right
            uy (float): the y-coordiante of the upper-right

        Returns:
            croppedDF (GeoPandas GeoDataFrame): A subset of the original df based
                on a spatial bounding box

        Raises:
            None

        Todo:
            * determine if this should overwrite the DataFrame

        Tests:
            None
        """
        croppedDF = self.df.cx[lx:ux,ly:uy]
        return croppedDF

    def burnDataFrameColumnToRaster(self,column,rasterPath,rasterCRS = 3857,rasterResolution=30,noDataValue=0):
        """ Writes a numeric vector of the dataframe to a raster

        Generates a GeoTIFF which contains a rasterized representation of the
        vector for a numeric column.  The raster will be sized as the boudning
        extents of the GeoDataFrame, with noValue's as specified by the user.

        Args:
            column (str): The column of the GeoDataFrame to rasterize
            rasterPath (str): Path to the output GeoTIFF
            rasterCRS (ENUM CRS): Projection of the output raster
            rasterResolution (float): The pixel size of the raster
            noDataValue (float): What to write where there is no data

        Return:
            None

        Todo:
            * Build this!  It'll be super useful!

        Tests:
            None
        """
        # 1. Check if the column is all numeric, and if not raise an error

        # 2. Get the bounds of the geometries

        # 3. Create an empty raster at those bounds

        # 4. Rasterize the vector to the empty raster

        # 5. Return a raster object
        return 0



class RasterLayer:
    """ Represents a raster

        Attributes:
            name (str): Neat name for the raster
            rasterPath (str): Filepath to the source GeoTIFF
            raster (GDAL Raster): Memory representation of the GDAL raster object
            crs (ENUM CRS): Projection of the raster
            scale (float): Resolution of the raster, in raster projection units
            minimum (float): Minimum value contained by the raster
            maximum (float): Maximum value contained by the raster
            unitType (str): Raster band unit types
            colorInterpretation (str): Raster band color interpretation
            colorTable (str): Raster band color table
            lx (float): x-coordinate of lower left, in same units as projection
            ly (float): y-coordinate of lower left, in same units as projection
            ux (float): x-coordinate of upper right, in same units as projection
            uy (float): y-coordinate of upper right, in same units as projection
            isCategorical (bool): True if the raster encodes continuous data,
                False if the raster encodes categorical data
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
        self.isCategorical = False


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
        """ Not yet implemented
        """
        return 0

    def export(self,newPath):
        """ Not yet implemented
        """
        self.rasterPath = newPath

    def reproject(self,crs=CRS.WMAS):
        """ Reprojects the raster into a tmp file and in memory

        Args:
            crs (ENUM CRS): target projection

        Returns:
            None

        Raises:
            None

        Tests:
            None
        """
        tmpRaster = "./tmp/tmp.tif"
        spatRef = osr.SpatialReference()
        spatRef.ImportFromEPSG(crs.value)
        gdal.Warp(tmpRaster,self.raster,dstSRS=spatRef)
        self.raster = gdal.Open(tmpRaster)
        self.crs = spatRef.ExportToWkt()
        self.rasterPath = "In memory: export to update"

    def crop(self,lx,ly,ux,uy):
        """ Not yet implemented
        """
        result = RasterLayer()
        return result

    def toPNG(self,outputPath):
        """ Produces PNG from the raster

        Currently uses hillshade and not the correct color table

        Args:
            outputPath (str): Filepath to write the PNG to

        Returns:
            argument (str): A string for debugging

        Raises:
            None

        Todo:
            * Implement correct color table

        Tests:
            None
        """
        argument = "gdaldem hillshade -of PNG %s %s" %(self.rasterPath,outputPath)
        cdArgument = "cd /home/noah/GIT/dissertation/results"
        os.system(cdArgument)
        os.system(argument)
        return argument


class Map:
    """ Wrapper for folium map to abstract out map product generation

        Helps to produce nice looking maps and load data

        Attributes:
            map (Folium Map): The map object
            name (str): Pretty-print version of the name for the map
    """

    def __init__(self,name="Not set"):
        self.map = folium.Map([37.7945, -92.1348], tiles='stamentoner', zoom_start=6)
        self.name = name

    def addRasterLayerAsOverlay(self,rasterLayer,opacity):
        """ Adds a raster object to the map

        Creates a temporary PNG from the rasterLayer at a single PNG path which
        is currently overwritten each time it is called

        Args:
            rasterLayer (RasterLayer): The raster object to be added to the map
            opacity (float): Allows transparency when producing the raster

        Returns:
            None

        Raises:
            None

        Todo:
            * Verify if raster projection is correct

        Tests:
            None
        """
        # 1. get boundary of raster
        bounds =[[rasterLayer.ly,rasterLayer.lx], [rasterLayer.uy,rasterLayer.ux]]

        # 2. export raster to png
        pngPath = "./tmp/temppng.png"
        rasterLayer.toPNG(pngPath)
        img = Image.open(pngPath)

        # 3. add ImageOverlay
        self.map.add_children(plugins.ImageOverlay(img,opacity=opacity,bounds=bounds))

    def addVectorLayerAsOverlay(self,vectorLayer):
        """ Adds a vector object to the map

        Loads each vector feature from the GeoDataFrame and adds to map

        Args:
            vectorLayer (GeoDataFrame): A GeoDataFrame populated with vectors

        Returns:
            None

        Raises:
            None

        Tests:
            None
        """
        gjson = vectorLayer.df.to_crs('3857').to_json()
        features = folium.features.GeoJson(gjson)
        self.map.add_children(features)


    def saveMap(self,filePath):
        """ Saves the map with all added data to an HTML file

        This file appears to run nicely on Firefox, with no added dependencies.
        Currently the basemap background is online, this should be investigated
        overall.

        Args:
            filePath (str): Filepath to save the file to, with .html extension

        Returns:
            None (produces map html at filePath)

        Raises:
            None

        Tests:
            None
        """
        self.map.save(filePath)
        print("Map saved to %s" %(filePath))

    def addCoolIcon(self,lat,lon,icon='bar-chart',popup='East London',color='blue'):
        """ Adds an icon from fontawesome.io

        Test code to help me build support for adding custom icons.  Currently
        reaches out to fontawesome.io for the icon, and requires a valid
        icon name

        Args:
            lat (float): Latitude of the marker
            lon (float): Longitude of the marker
            icon (str): A valid fontawesome.io icon type
            popup (str): Popup text
            color (str): The color of the popup, need to verify valid color values

        Returns:
            None

        Raises:
            None

        Todo:
            * Add support for projecting the point
            * Allow the user to change the icon
            * Allow the user to change the popup, for instance to HTML
            * Make sure None popups do not crash it
        Tests:
            >>> addCoolIcon(35,-91,icon='bicycle',color='red')
        """
        coolIcon = folium.Marker([lat,lon],
              popup=popup,
              icon=folium.Icon(color=color,icon=icon, prefix='fa')
             )
        self.map.add_child(coolIcon)

    def test_generateRandomLatLonPair(self,latMin,latMax,lonMin,lonMax):
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

    def test_addTimeSeriesHeatMap(self,latMin=35,latMax=50,lonMin=-90,lonMax=-80):
        """ Creatse a random time series map to demonstrate functionality

        This is test code to demonstrate the ability to add time series, which
        will be a cool way to demonstrate generations in genetic algorithms

        Args:
            None

        Returns:
            None

        Raises:
            None

        Todo:
            * Have this take in a dataframe and value of the time column instead

        Tests:
            None
        """
        heat_data = []
        for gen in range(0,5):
            gen = []
            for i in range(0,1001):
                lat,lon = self.test_generateRandomLatLonPair(latMin,latMax,lonMin,lonMax)
                #lat = np.random.randint(35,50)
                #lon = np.random.randint(-90,-80)
                val = [lat,lon]
                gen.append(val)
            heat_data.append(gen)
        hm = plugins.HeatMapWithTime(heat_data)
        self.map.add_child(hm)

    def addTimeSeriesHeatMapFromArray(self,heat_data):
        hm = plugins.HeatMapWithTime(heat_data)
        self.map.add_child(hm)


## FUNCTIONS
def createEmptyRaster(rasterPath,topLeftX,topLeftY,cellSize,width,height,epsg,dtype=np.uint32):
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
        dtype (dtype): The data type of the raster

    Returns:
        rasterPath (str): The path of the created raster, as a confirmation that
            the raster was written

    Raises:
        None

    Tests:
        <<< raster[25:50,25:50] = 100 # this code is for testing, and not a unit test
    """
    gdal_dtype = gdal.GDT_UInt32
    if dtype == np.byte:
        gdal_dtype = gdal.GDT_Byte
    elif dtype == np.int16:
        gdal_dtype = gdal.GDT_Int16
    elif dtype == np.int32:
        gdal_dtype = gdal.GDT_Int32
    elif dtype == np.int64:
        dtype = np.int32 # This is a kludge
        gdal_dtype = gdal.GDT_Int32
    elif dtype == np.uint16:
        gdal_dtype = gdal.GDT_UInt16
    elif dtype == np.uint32:
        gdal_dtype = gdal.GDT_UInt32
    elif dtype == np.float32:
        gdal_dtype = gdal.GDT_Float32
    elif dtype == np.float64:
        gdal_dtype = gdal.GDT_Float64
    else: # Fallback condition
        dtype = np.uint32
        gdal_dtype = gdal.GDT_UInt32

    geotransform = [topLeftX,cellSize,0,topLeftY,0,-cellSize]
    driver = gdal.GetDriverByName("GTiff")
    dst_ds = driver.Create(rasterPath, width, height, 1, gdal_dtype )
    dst_ds.SetGeoTransform(geotransform)
    srs = osr.SpatialReference()
    srs.ImportFromEPSG(epsg)
    dst_ds.SetProjection(srs.ExportToWkt())
    raster = np.zeros((height,width),dtype=dtype)
    dst_ds.GetRasterBand(1).WriteArray(raster)
    return rasterPath

def rasterizeGeodataFrameColumn(df,column,outputRasterPath,resolution=30,crs=None,noDataValue=0):
    """ Creates a raster from the specified column

    Given a GeoDataFrame and numeric column, creates a raster with the extents
    as the overall dataframe and specified resolution, in the same CRS as the
    dataframe unless CRS is specified.

    Args:
        df (GeoPandas GeoDataFrame): The collection of vector features
        column (str): The name of a numeric column in the df, to be rasterized
        rasterPath (str): Where to place the newly created raster
        resolution (float): The cell size of the output raster
        crs (ENUM CRS): The target projection.  If None, same as df.
        noDataValue (int/float): The value to write where no data exists

    Returns:
        rasterPath (str): The path of the created raster, as a confirmation that
            the raster was written

    Raises:
        None

    Todo:
        * Raise if column is not numeric
        * Implement re-projection

    Tests:
        >>> aoiDF = gpd.read_file("./test_data/geojson.json")
        >>> aoiDF.CRS = {'init':'epsg:4326'}
        >>> aoiDF = aoiDF.to_crs({'init':'epsg:3857'})
        >>> aoiPolygon = aoiDF.geometry[0]
        >>> gridSpacing = 30
        >>> squareList = []
        >>> bounds = aoiPolygon.bounds
        >>> ll = bounds[:2]
        >>> ur = bounds[2:]
        >>> for x in floatrange(ll[0],ur[0],gridSpacing):
        >>>    for y in floatrange(ll[1],ur[1],gridSpacing):
        >>>        square = Polygon([[x,y],[x+gridSpacing,y],[x+gridSpacing,y+gridSpacing],[x,y+gridSpacing]])
        >>>        if square.within(aoiPolygon):
        >>>            squareList.append(square)
        >>> df = gpd.GeoDataFrame(squareList)
        >>> df.columns = ['geometry']
        >>> df['score'] = np.random.randint(0,100,len(df.index))
        >>> rp = rasterizeGeodataFrameColumn(df,'score','./results/testrasterizationfunction.tif')
        >>> rp = rasterizeGeodataFrameColumn(df,'score','./results/testrasterizationfunctionNegativeNoData.tif',noDataValue=-1)
    """
    # 1. Check if the column is all numeric, and if not raise an error.  Also,
    #       determine the data type and make sure all of the rasters reflect
    dtype = df[column].dtype.type # type numpy.dtype
    if dtype == np.int16:
        pass
    elif dtype == np.int32:
        pass
    elif dtype == np.int64:
        dtype = np.int32
    elif dtype == np.uint8:
        dtype = np.uint16
    elif dtype == np.uint16:
        pass
    elif dtype == np.uint32:
        pass
    elif dtype == np.uint64:
        dtype = np.uint32
    elif dtype == np.float32:
        pass
    elif dtype == np.float64:
        pass
    else: # Fallback condition
        dtype = np.uint32
        gdal_dtype = gdal.GDT_UInt32

    # 2. Get the bounds of the geometries
    lx,ly,ux,uy = df.total_bounds
    width = int(np.ceil((ux-lx)/resolution))
    height = int(np.ceil((uy-ly)/resolution))

    # 3. Create an empty raster at those bounds
    # https://gis.stackexchange.com/questions/31568/gdal-rasterizelayer-does-not-burn-all-polygons-to-raster
    rasterPath = "./results/tmpRasterForDF.tif"
    rasterPath = createEmptyRaster(rasterPath,lx,uy,resolution,width,height,3857,dtype=dtype)

    # 4. Rasterize the vector to a copy of the empty raster
    rst = rasterio.open( rasterPath )
    meta = rst.meta
    meta.update( compress='lzw' )
    start = datetime.datetime.now()
    with rasterio.open( outputRasterPath, 'w', **meta ) as out:
        out_arr = out.read( 1 )
        # this is where we create a generator of geom, value pairs to use in rasterizing
        shapes = ( (geom,value) for geom, value in zip( df.geometry, df[column] ) )
        #burned = features.rasterize( shapes=shapes, fill=0, out=out_arr, transform=out.transform,dtype=rasterio.float32)

        burned = features.rasterize( shapes=shapes, fill=noDataValue, out=out_arr, transform=out.transform,dtype=dtype)

        out.write_band( 1, burned )
    stop = datetime.datetime.now()
    timeDelta = stop - start
    print "Raster created with datatype %s in %s seconds at %s" %(dtype.name,timeDelta.seconds,outputRasterPath)
    return outputRasterPath


def ensiteLayertoDF(layerID):
    con = psycopg2.connect(database="ensite", user="postgres",password="postgres",host="127.0.0.1")
    cur = con.cursor()
    queryStatement = "SELECT ensite_study_id,projection,name, geometry_type FROM ensite_layer WHERE id = %s;" %(layerID) # todo, make sure studyID remains an integer
    cur.execute(queryStatement)
    for row in cur:
        #do something with every single row here
        #optionally print the row
        print row

def dataFrameToENSITEDatabase(df,studyID,layerName,layerID=None,geometryType = "Polygon"):
    """ Writes a Vector GeoDataFrame into the ENSITE database

    This code is mirrored from the ENSITE ImportExportLibrary, and is necessary
    to allow site search to directly insert into the database.  Site Search
    results will be written to the database as a single vector layer per
    study objective.

    Args:
        df (GeoPandas GeoDataFrame): A vector GeoDataFrame with CRS set
        studyID (int): The ID for the ENSITE study this layer belongs to.
        layerName (str): The pretty-print name to refer to this layer as
        layerID (int): Passed if a layer has already been creaed by the ENSITE
            user interface, otherwise this is sest during insertion
        geometryType (str): Right now used as a kludge to make sure the layer ends
            up in the correct database in ENSITE

    Returns:
        None

    Raises:
        None

    Todo:
        * Lots of testing
        * Evaluate security

    Tests:
        None
    """
    con = psycopg2.connect(database="ensite", user="postgres",password="postgres",host="127.0.0.1")
    cur = con.cursor()
    # get the projection of the df
    try:
		projection = df.crs['init']
    except:
		projection = "epsg:3857" # kludge


    # insert the layer
    #geometryType = df.loc[0]["geometry"].type #TODO try catch type thing
    #geometryType = "Polygon"

    if not layerID:
        insertStatement = "INSERT INTO ensite_layer (ensite_study_id,projection,name, geometry_type,primary_color,secondary_color) VALUES (%s,'%s','%s', '%s','rgba(0,0,0,.6)', 'rgba(150,150,150,.25)') RETURNING id;" %(studyID,projection,layerName, geometryType) # todo, make sure studyID remains an integer
        cur.execute(insertStatement)
        layerID = cur.fetchone()[0]


    # set up the columns
    columns = [column for column in df.columns if column != "geometry"]
    columnTypes = {}
    for column in columns:
        dtype = df[column].dtype.name
        if dtype == 'object':
            dtype = 'text'
        elif dtype =='float64':
            dtype = 'real'
        columnTypes[column] = dtype
    columnKeys = {}
    for columnName,columnType in columnTypes.iteritems():
        insertStatement = "INSERT INTO ensite_feature_property_name (name,type,ensite_layer_id) VALUES ('%s','%s',%s) RETURNING id;" %(columnName,columnType,layerID)
        cur.execute(insertStatement)
        columnID = cur.fetchone()[0]
        columnKeys[columnName] = columnID
        #con.commit()

    # insert the features and their properties one at a time
    for i,row in df.iterrows():
        # write the feature
        geometry = row["geometry"]
        geometryType = geometry.type

        geometryTable = "ensite_feature_point"
        if geometryType != "Point":
            geometryTable = "ensite_feature_vector"
        insertStatement = "INSERT INTO ensite_feature (ensite_layer_id,type) VALUES (%s,'%s') RETURNING id;" %(layerID,geometryTable)
        cur.execute(insertStatement)
        featureID = cur.fetchone()[0]
        #con.commit()

        # write the feature geometry
        geometry_wkt = geometry.wkt
        insertStatement = "INSERT INTO %s (ensite_feature_id,geometry) VALUES (%s,ST_GeometryFromText('%s',%s));" %(geometryTable,featureID,geometry_wkt,projection.split(':')[1])
#        insertStatement = "INSERT INTO feature_point (feature_id,geometry) VALUES (%s,ST_SetSRID(%s::geometry,%s) RETURNING id" %(featureID,geometry_wkb,4326)
        cur.execute(insertStatement)
        #con.commit()

        for columnName,columnKey in columnKeys.iteritems():
            # write the properties
            valueType = columnTypes[columnName]
            columnID = columnKeys[columnName]
            columnValue = row[columnName]
            # clean up columnValue for commit, courtesy of http://stackoverflow.com/questions/3224268/python-unicode-encode-error
            if valueType == 'text':
                columnValue = unicode(columnValue)
            else:
                columnValue = unicode(str(columnValue))
            if columnValue:
                insertStatment = "INSERT INTO ensite_feature_property_value (ensite_feature_id,ensite_feature_property_name_id,value,value_type) VALUES (%s,%s,'%s','%s');" %(featureID,columnID,columnValue.replace("'",""),valueType)
            else:
                insertStatment = "INSERT INTO ensite_feature_property_value (ensite_feature_id,ensite_feature_property_name_id,value,value_type) VALUES (%s,%s,%s,'%s');" %(featureID,columnID,"NULL",valueType)
            cur.execute(insertStatment)
            #con.commit()

    con.commit()
    return layerID

def loadFGDB(fgdbPath):
    fgdbPath = "C:/Users/RDCERNWG/Documents/GIT/FLW_Missouri Mission Folder/RECON/enfire.gdb"
    with fiona.open(fgdbPath, driver="OpenFileGDB") as src:
        return 0


## CURRENT TEST
def tryingToLearnToMapWIthTables():
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
    map.saveMap("./results/testHeatMapWithTime_homeDesktop.html")

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
