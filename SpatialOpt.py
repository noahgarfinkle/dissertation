# -*- coding: utf-8 -*-
"""
Manages simple objective functions for vector data layers
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

import CandidateDataFrameOperations as candidates
import ENSITEIO as eio
import Objective_Analytic as objective_analytic
import Objective_Raster as objective_raster
import pgdissroute as pgdissroute
import SpatialIO as io
import SpatialOpt as opt
