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

import CandidateDataFrameOperations as candidates
import Objective_Analytic as objective_analytic
import Objective_Raster as objective_raster
import Objective_Vector as objective_vector
import pgdissroute as pgdissroute
import SpatialIO as io
import SpatialOpt as opt

def dfFromPostGIS(layerID):
    con = psycopg2.connect(database="ensite", user="postgres",password="postgres",host="127.0.0.1")
    cur = con.cursor()
    # get the layer information
    geometryTable = "ensite_feature_point"
    queryStatement = "SELECT ensite_study_id,projection,name, geometry_type FROM ensite_layer WHERE id = %s;" %(layerID) # todo, make sure studyID remains an integer
    cur.execute(queryStatement)
    for row in cur:
        ensite_study_id = row[0]
        projection = row[1]
        name = row[2]
        geometry_type = row[3]
        if geometry_type == "Point":
            pass
        elif geometry_type == "Raster":
            pass
        else:
            geometryTable = "ensite_feature_vector"


    # get the property information
    columns = {}
    columnIDs = {}
    columnData = {}
    queryStatement = "SELECT id,name,type from ensite_feature_property_name WHERE ensite_layer_id = %s;" %(layerID)
    cur.execute(queryStatement)
    for row in cur:
        columnID = row[0]
        columnName = row[1]
        columnType = row[2]
        columns[columnName] = columnType
        columnIDs[columnID] = columnName
        columnData[columnName] = {}

    # get the features
    featureIDs = []
    queryStatement = "SELECT id FROM ensite_feature WHERE ensite_layer_id = %s;" %(layerID)
    cur.execute(queryStatement)
    for row in cur:
        featureID = row[0]
        featureIDs.append(featureID)

    # get the geometries
    geometries = {}
    for featureID in featureIDs:
        queryStatement = "SELECT ST_AsText(geometry) FROM %s WHERE ensite_feature_id = %s;" %(geometryTable,featureID)
        cur.execute(queryStatement)
        for row in cur:
            row_geometry = row[0]
            geometries[featureID] = row_geometry
            #print row_geometry

    # get the properties
        for columnID in columnIDs.iterkeys():
            columnName = columnIDs[columnID]
            queryStatement = "SELECT value,value_type FROM ensite_feature_property_value WHERE ensite_feature_id= %s AND ensite_feature_property_name_id = %s;" %(featureID,columnID)
            cur.execute(queryStatement)
            count = 0
            for row in cur:
                #print row
                property_value = row[0]
                property_type = row[1]
                columnData[columnName][featureID] = property_value
                count += 1
            if count == 0:
                columnData[columnName][featureID] = "None"

    df = gpd.GeoDataFrame(columns=columns.keys())
    df.geometry = [loads(x) for x in geometries.itervalues()]
    for k,v in columnData.iteritems():
        df[k] = v.values()

    df.crs = {'init':'EPSG:%s' %(projection.split(':')[1])}
    return df

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

def writeDataFrameToENSITEDB(df,studyID,layerName,layerID=None):
    """ Writes results into ENSITE database

        Evaluation function

    Args:
        df (GeoPandas GeoDataFrame): Set of candidate solutions
        studyID (int): ID of ENITE study the layer should belong to
        layerName (str): Pretty-print name to display in ENSITE
        layerID (int): If None, a new layer is created.  If provided, the data
            is added to an existing layer

    Returns:
        layerID (int): The returned layer id in the database

    Raises:
        None

    Todo:
        * Currently rewritten to only write polygon layers

    Tests:
        None
    """
    layerID = io.dataFrameToENSITEDatabase(df,studyID,layerName,layerID=layerID)
    return layerID
