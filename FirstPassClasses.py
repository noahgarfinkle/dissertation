coding: utf-8 -*-
"""
Parses an XML file in order to evaluate sites
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
__date_created__ = "10 FEBRUARY 2018"

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

import SpatialIO as io

## HELPFUL FOR DEBUGGING
# %matplotlib inline
# pd.options.display.max_columns = 300

## SETUP
validStats = utils.VALID_STATS

## CLASSES
class Input:
    """ Python data structure for parsing the XML data structure

        Attributes and methods for parsing the XML input file, ensuring that first
        pass is usable across a range of applications

        Attributes:
            xmlPath (str): Path to the XML file
            tree (XML tree): Python representation of the XML file using lxml etree
            root (XML tree element): Root node of the lxml etree
            siteSearches (list: SiteSearch): The study objectives
            siteRelationalConstraints (list: SiteRelationalConstraint): The relationships
                between the study objectives
            resultDir (str): Base path for writing results to
            studyObjectiveID (long): Study objectiveve for reference to PostGIS database
                object
    """

    def __init__(self,xmlPath):
        self.xmlPath = xmlPath
        self.tree = ET.parse(xmlPath)
        self.root = tree.getroot()
        self.siteSearches = self.retrieveSiteSearches()
        self.siteRelationalConstraints = self.retrieveSiteRelationalConstraints()
        self.resultDir = root.attrib['resultDir']
        self.studyObjectiveID = root.attrib['studyObjectiveID']

    def pretty_print(self):
        print ET.tostring(self.root,pretty_print=True)

    def retrieveSiteSearches(self):
        """ Converts each XML study objective to type SiteSearch

        Retreives each indivdidual study objective, such as an individual airfield, and
        appends to a list

        Args:
            None

        Returns:
            None (updates siteSearches)

        Raises:
            None

        Tests:
            None
        """
        siteSearches = []
        siteSearchesElement = self.root[0]
        for siteSearchElement in siteSearchesElement:
            print "%s: %s" %(siteSearchElement.tag, siteSearchElement.attrib)
            for siteSearchElementChild in siteSearchElement:
                print "\t%s: %s" %(siteSearchElementChild.tag,siteSearchElementChild.attrib)
            siteSearch = SiteSearch()
            siteSearches.append(siteSearch)
        return siteSearches

    def retrieveSiteRelationalConstraints(self):
        """ Converts each XML relational constraint to type SiteRelationalConstraint

        Captures the interelationships between study objectives, linking the objects

        Args:
            None

        Returns:
            None (updates siteRelationalConstraints)

        Raises:
            None

        Tests:
            None
        """
        siteRelationalConstraints = self.root[1]
        siteRelationalConstraint = SiteRelationalConstraint()
        siteSearchRelationalConstraints = []
        siteRelationalConstraints.append(siteRelationalConstraint)
        return siteSearchRelationalConstraints
