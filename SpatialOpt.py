# -*- coding: utf-8 -*-
"""
Implements the spatial optimization algorithms
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
__date_created__ = "12 May 2018"

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
import shapely.affinity
from rasterstats import zonal_stats, raster_stats, point_query, utils
import matplotlib.pyplot as plt
import datetime
import time

import CandidateDataFrameOperations as candidates
import ENSITEIO as eio
import Objective_Analytic as objective_analytic
import Objective_Raster as objective_raster
import pgdissroute as pgdissroute
import SpatialIO as io

## OBJECTIVE FUNCTIONS BETWEEN CANDIDATES
def evaluateCandidates_EuclideanDistance(df1,index1,df2,index2):
    geom1 = df1[index1:index1+1]
    geom1.crs = {'init':'EPSG:3857'}

    geom2 = df2[index2:index2+1]
    geom2.crs = {'init':'EPSG:3857'}

    euclideanDistance = geom2.distance(geom1.geometry[index1]).min()
    return euclideanDistance

def evaluateCandidates_DrivingDistance(df1,index1,df2,index2):
    geom1 = df1[index1:index1+1]
    geom1.crs = {'init':'EPSG:3857'}
    geom1 = geom1.to_crs({'init':'EPSG:4326'})
    geom1 = geom1['geometry']

    geom2 = df2[index2:index2+1]
    geom2.crs = {'init':'EPSG:3857'}
    geom2 = geom2.to_crs({'init':'EPSG:4326'})
    geom2 = geom2['geometry']

    # reproject the geometries
    totalDriveDistance = sitesearch.pgdissroute.calculateRouteDistance(geom1.centroid.x[index1],geom1.centroid.y[index1],geom2.centroid.x[index2],geom2.centroid.y[index2])
    return totalDriveDistance

## OBJECTIVE FUNCTIONS BETWEEN A CANDIDATE AND SOURCE
