# -*- coding: utf-8 -*-
"""
Generates and maintains GeoPandas GeoDataFrames which help to organize
candidate sites for site serach.
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
__date_created__ = "10 May 2018"

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

import ENSITEIO as eio
reload(eio)
import FirstPass as firstpass
reload(firstpass)
import Objective_Analytic as objective_analytic
reload(objective_analytic)
import Objective_Raster as objective_raster
reload(objective_raster)
import Objective_Vector as objective_vector
reload(objective_vector)
import pgdissroute as pgdissroute
reload(pgdissroute)
import SpatialIO as io
reload(io)
import SpatialOpt as opt
reload(opt)

## WKT MANAGEMENT
def projectWKT(wkt,from_epsg,to_epsg):
    """ Reprojects a string of well known text


    Args:
        wkt (str): A string of well-known-text
        from_epsg (int): The current projection of wkt
        to_epsg (int): The desired projection of wkt

    Returns:
        reprojectedWKT (str): well-known-text projected into to_epsg

    Raises:
        None

    Tests:
        None
    """
    feat = loads(wkt)
    df_to_project = gpd.GeoDataFrame([feat])
    df_to_project.columns = ['geometry']
    df_to_project.crs = {'init':'EPSG:' + str(from_epsg)}
    df_to_project = df_to_project.to_crs({'init':'EPSG:' + str(to_epsg)})
    reprojectedWKT = df_to_project.geometry[0].to_wkt()
    return reprojectedWKT

## GRID FUNCTIONS
def floatrange(start, stop, step):
    """ Generates a range between two floats with a float step size

    Taken from http://portolan.leaffan.net/creating-sample-points-with-ogr-and-shapely-pt-2-regular-grid-sampling/

    Args:
        start (float): Lower bound, inclusive
        stop (float): Upper bound, exclusive
        step (float): Step size

    Returns:
        generator (generator of floats): Range between start and stop

    Raises:
        None

    Tests:
        None
    """
    while start < stop:
        yield start
        start += step

def polygonBuilder(aoiPolygon, epsg="3857", wkt="POLYGON ((0 0, 91 0, 91 1700, 0 1700, 0 0))",
                    units="m", gridSpacing="800", rotationUnits="degrees",
                    rotationStart="0", rotationStop="180", rotationSpacing="90"):
    """ Produces a GeoPandas GeoDataFrame of arbitrary polygons on a test grid

    This produces the fundamental solution data structure of First Pass, a
    GeoPandas GeoDataFrame sized and rotated as specified, and falling exclusively
    within the aoiPolygon, ideally which has already had no-build areas removed.
    Can take arbitrary polygon definitions in WKT.

    Args:
        aoiPolygon (Shapely Polygon): A representation of the area of interest,
            ideally with holes removed for areas which should not fall within
            candidate solutions
        epsg (int): The projection of the aoiPolygon
        wkt (str): A well-known-text representation of the polygon being built
        units (str): The distance units to utilize, typically 'm' for meters
        gridSpacing (float): Distance between lower left corners of evaluation sites,
            may be replaced in the future with centroids once PolygonBuilder is
            implemented.  Converted to float.
        rotationUnits (str): Defines how rotation is treated, typically degrees
        rotationStart (float): Starting rotation, with 0 defined as due North,
            and incrementing clockwise
        rotationStop (float): Ending rotation, with 0 defined as due North,
            and incrementing clockwise
        rotationSpacing (float): Increment of rotation to evaluate

    Returns:
        evaluationDF (GeoPandas GeoDataFrame): Each row represents a candidate
        site for evaluation

    Raises:
        None

    Todo:
        * Implement units and rotationUnits

    Tests:
        None
    """
    epsg = int(epsg)
    gridSpacing=float(gridSpacing)
    rotationStart=float(rotationStart)
    rotationStop=float(rotationStop)
    rotationSpacing=float(rotationSpacing)

    # build the polygon template
    template = loads(wkt)
    templateCentroid = template.centroid

    # loop over the grid and rotations
    candidatesList = []
    bounds = aoiPolygon.bounds
    ll = bounds[:2]
    ur = bounds[2:]
    # https://stackoverflow.com/questions/30457089/how-to-create-a-polygon-given-its-point-vertices
    start = datetime.datetime.now()
    for x in floatrange(ll[0],ur[0],gridSpacing):
        for y in floatrange(ll[1],ur[1],gridSpacing):
            xoffset = x - templateCentroid.x
            yoffset = y-  templateCentroid.y
            candidate = shapely.affinity.translate(template,xoff=xoffset,yoff=yoffset)
            for rotation in floatrange(rotationStart,rotationStop+1,rotationSpacing):
                rotatedCandidate = shapely.affinity.rotate(candidate, -rotation, origin='centroid', use_radians=False)
                if rotatedCandidate.within(aoiPolygon):
                    candidatesList.append(rotatedCandidate)
    end = datetime.datetime.now()
    timeElapsed = end - start
    nFeatures = len(candidatesList)
    print "Generated %s candidate polygons in %s seconds" %(nFeatures,timeElapsed.seconds)
    evaluationDF = gpd.GeoDataFrame(candidatesList)
    evaluationDF.columns = ['geometry']
    return evaluationDF

def buildGriddedSearchFromXML(siteConfiguration,searchParameters):
    """ ENSITE MSSPIX gridded search builder

    Builds the evaluation DF for a gridded search with unrpojected WKT AOI

    Args:
        siteConfiguration (lxml): Definition of site to be searched for
        searchParameters (lxml): Definition of AOI and gridded search behavior

    Returns:
        evaluationDF (GeoPandas GeoDataFrame): A set of polygons based on the
            template wkt in siteConfiguration, falling within the AOI wkt
            in searchParameters, on a grid and rotation as specified in
            searchParameters

    Raises:
        None

    Tests:
        None
    """
    print "Gridded Search"

    siteConfiguration_WKT = siteConfiguration.attrib['wkt']
    siteConfiguration_Units = siteConfiguration.attrib['units']

    aoi_WKT = searchParameters.attrib['wkt']
    aoi_units = searchParameters.attrib['units']
    aoi_EPSG = searchParameters.attrib['wkt_epsg']
    # Reproject aoiPolygon
    if aoi_EPSG != "3857":
        aoi_WKT = projectWKT(aoi_WKT,aoi_EPSG,3857)
    aoiPolygon = loads(aoi_WKT)

    gridSpacing = searchParameters.attrib['gridSpacing']
    rotationStart = searchParameters.attrib['rotationStart']
    rotationStop = searchParameters.attrib['rotationStop']
    rotationSpacing = searchParameters.attrib['rotationSpacing']

    evaluationDF = polygonBuilder(aoiPolygon,gridSpacing=gridSpacing,rotationStart=rotationStart,rotationStop=rotationStop,rotationSpacing=rotationSpacing, wkt=siteConfiguration_WKT)

    return evaluationDF

def buildSingleSiteSearchFromXML(siteConfiguration,searchParameters):
    print "Single Site Search"

    siteConfiguration_WKT = siteConfiguration.attrib['wkt']
    siteConfiguration_Units = siteConfiguration.attrib['units']

    aoi_WKT = searchParameters.attrib['wkt']
    aoi_units = searchParameters.attrib['units']
    aoi_EPSG = searchParameters.attrib['wkt_epsg']
    # Reproject aoiPolygon
    if aoi_EPSG != "3857":
        aoi_WKT = projectWKT(aoi_WKT,aoi_EPSG,3857)
    print aoi_WKT
    aoiPolygon = loads(aoi_WKT)
    aoiPolygonList = [aoiPolygon]



    evaluationDF = gpd.GeoDataFrame(aoiPolygonList)
    evaluationDF.columns = ['geometry']

    return evaluationDF
