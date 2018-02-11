# -*- coding: utf-8 -*-
"""Parses an XML file in order to evaluate sites
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
__date_created__ = "10 FEBRUARY 2018"

# IMPORTS
from lxml import etree as ET
import gdaltools as gdt
from osgeo import gdal, osr, ogr
import geopandas as gpd
import fiona
import rasterio
import numpy as np
import matplotlib.pyplot as plt

""" REFERENCES
http://lxml.de/tutorial.html
https://mapbox.s3.amazonaws.com/playground/perrygeo/rasterio-docs/cookbook.html#rasterizing-geojson-features
"""


# CLASSES
class SiteSearch:
    def __init__(self):
        self.crs = None
        self.AreaOfInterest = None

class SiteSuitabilityCriteria:
    def __init__(self):
        return None

class SiteSuitabilityCriteriaEvaluahttps://mapbox.s3.amazonaws.com/playground/perrygeo/rasterio-docs/cookbook.html#rasterizing-geojson-featurestion:
    def __init__(self):EvalFallBack
        return None

class SiteRelationalConstraint:
    def __init__(self):
        return None

class Input:
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
        siteRelationalConstraints = self.root[1]
        siteRelationalConstraint = SiteRelationalConstraint()
        siteSearchRelationalConstraints = []
        siteRelationalConstraints.append(siteRelationalConstraint)
        return siteSearchRelationalConstraints

# SPATIAL FUNCTIONS
def distance():
    distances = gdal.distances()

    return None

def creatQAFRaster(crs,lx,ly,ux,uy,qafCellSize):
    numberXCells = np.ceiling((ux-lx)/qafCellSize)
    numberYCells = np.ceiling((uy-ly)/qafCellSize)

    newUX = lx + (numberXCells * qafCellSize)
    newUY = ly + (numberYCells * qafCellSize)

    qafMatrix = np.empty([numberXCells,numberYCells])

    """
  nx <- ceiling((ux - lx) / qafCellSize)
  ny <- ceiling((uy - ly) / qafCellSize)

  ux <- lx + (nx * qafCellSize)
  uy <- ly + (ny * qafCellSize)

  qafMatrix <- matrix(0,ny,nx)
  qafRaster <- raster(qafMatrix)
  extent(qafRaster) <- c(lx,ux,ly,uy)

  projection(qafRaster) <- CRS('+init=epsg:3857') # wgs 1984 web mercator auxillary sphere
    """

def createQAFGridSurface():
    return 0

# TESTS
xmlPath = "./input.xml"

# test distance
raster_path = "/home/noah/FLW_Missouri Mission Folder/RASTER/DEM_CMB_ELV_SRTMVF2.tif"
raster = gdal.Open(raster_path)
raster_array = raster.ReadAsArray()
raster_array
%matploblib inline
plt.imshow(raster_array)
vector_path = "/home/noah/FLW_Missouri Mission Folder/VECTOR/TransportationGroundChttps://mapbox.s3.amazonaws.com/playground/perrygeo/rasterio-docs/cookbook.html#rasterizing-geojson-featuresrv.shp"
driver = ogr.GetDriverByName('ESRI Shapefile')
lines = driver.Open(vector_path,0)
linesLayer = lines.GetLayer()

# rasterize the vector
# https://github.com/mapplus/qgis-scripts/blob/master/scripts/Raster%20Euclidean%20Distance%20Analysis.py


rasterizedLinesLayer = gdal.RasterizeLayer(linesLayer)
