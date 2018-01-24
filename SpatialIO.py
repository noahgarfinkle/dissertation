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
Images can be saved as new files using the save() function. The resulting file will have all the same metadata and number of bands, but with all processing applied. The datatype of the new image will be the same as the old one unless the dtype keyword is provided. Note that providing the dtype keyword does not scale the values however, it is up to the user to scale values to the desired range to match the output file created. Use the GeoImage.autoscale() function to automatically scale all bands, or use the GeoRaster.scale() function on each band to specify the input and output ranges.
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



def readColorTable(color_file):
    '''
    The method for reading the color file.
    * If alpha is not defined, a 255 value is set (no transparency).
    '''

    color_table = {}
    if color_file != None:
        if exists(color_file) is False:
            raise Exception("Color file " + color_file + " does not exist")

        fp = open(color_file, "r")
        for line in fp:
            if line.find('#') == -1 and line.find('/') == -1:
                entry = line.split()
                if len(entry) == 5:
                    alpha = int(entry[4])
                else:
                    alpha=255
                color_table[eval(entry[0])]=[int(entry[1]),int(entry[2]),int(entry[3]),alpha]
        fp.close()
    else:
        color_table[127] = [255,0,0,0.3]
        color_table[150] = [0,255,0,0.3]
        color_table[200] = [0,0,255,0.3]
        color_table[240] = [192,192,192,0.3]

    return color_table


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
        outputPath = "./results/testout3.png"
        argument = "gdaldem hillshade -of PNG %s %s" %(self.rasterPath,outputPath)
        cdArgument = "cd /home/noah/GIT/dissertation/results"
        os.system(cdArgument)
        os.system(argument)
        return argument

    def toPNG(self, out_file_name, color_file=None , raster_band=1, discrete=True):
        # http://geoexamples.blogspot.com/2013/06/gdal-performance-ii-create-png-from.html
        # http://geoexamples.blogspot.com/2012/02/colorize-raster-with-gdal-python.html

        dataset = self.raster
        if dataset == None:
            raise Exception("Unable to read the data file")

        band = dataset.GetRasterBand(raster_band)

        block_sizes = band.GetBlockSize()
        x_block_size = block_sizes[0]
        y_block_size = block_sizes[1]

        xsize = band.XSize
        ysize = band.YSize

        max_value = band.GetMaximum()
        min_value = band.GetMinimum()

        if max_value == None or min_value == None:
            stats = band.GetStatistics(0, 1)
            max_value = stats[1]
            min_value = stats[0]

        #Reading the color table
        color_table = readColorTable(color_file)
        #Adding an extra value to avoid problems with the last & first entry
        if sorted(color_table.keys())[0] > min_value:
            color_table[min_value - 1] = color_table[sorted(color_table.keys())[0]]

        if sorted(color_table.keys())[-1] < max_value:
            color_table[max_value + 1] = color_table[sorted(color_table.keys())[-1]]
        #Preparing the color table and the output file
        classification_values = color_table.keys()
        classification_values.sort()


        rgb = zeros((ysize, xsize, 4), dtype = uint8)

        for i in range(0, ysize, y_block_size):
            if i + y_block_size < ysize:
                rows = y_block_size
            else:
                rows = ysize - i

            for j in range(0, xsize, x_block_size):
                if j + x_block_size < xsize:
                    cols = x_block_size
                else:
                    cols = xsize - j


                values = band.ReadAsArray(j, i, cols, rows)
                r = zeros((rows, cols), dtype = uint8)
                g = zeros((rows, cols), dtype = uint8)
                b = zeros((rows, cols), dtype = uint8)
                a = zeros((rows, cols), dtype = uint8)

                for k in range(len(classification_values) - 1):
                    #print classification_values[k]
                    if classification_values[k] < max_value and (classification_values[k + 1] > min_value ):
                        mask = logical_and(values >= classification_values[k], values < classification_values[k + 1])
                        if discrete == True:
                            r = r + color_table[classification_values[k]][0] * mask
                            g = g + color_table[classification_values[k]][1] * mask
                            b = b + color_table[classification_values[k]][2] * mask
                            a = a + color_table[classification_values[k]][3] * mask
                        else:
                            v0 = float(classification_values[k])
                            v1 = float(classification_values[k + 1])

                            r = r + mask * (color_table[classification_values[k]][0] + (values - v0)*(color_table[classification_values[k + 1]][0] - color_table[classification_values[k]][0])/(v1-v0) )
                            g = g + mask * (color_table[classification_values[k]][1] + (values - v0)*(color_table[classification_values[k + 1]][1] - color_table[classification_values[k]][1])/(v1-v0) )
                            b = b + mask * (color_table[classification_values[k]][2] + (values - v0)*(color_table[classification_values[k + 1]][2] - color_table[classification_values[k]][2])/(v1-v0) )
                            a = a + mask * (color_table[classification_values[k]][3] + (values - v0)*(color_table[classification_values[k + 1]][3] - color_table[classification_values[k]][3])/(v1-v0) )

                rgb[i:i+rows,j:j+cols, 0] = r
                rgb[i:i+rows,j:j+cols, 1] = g
                rgb[i:i+rows,j:j+cols, 2] = b
                rgb[i:i+rows,j:j+cols, 3] = a

        scipy.misc.imsave(out_file_name, rgb)



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


# test composite

map = Map(name="test map")
rl = RasterLayer(name="test raster")
rl.from_file("/home/noah/GIT/dissertation/test_data/testelevunproj.tif")
vl = VectorLayer(name="test vector")
rl.toPNG("./results/testout.png")

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
