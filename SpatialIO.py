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


from enum import Enum
class CRS(Enum):
    WGS84 = 4326
    WMAS = 3857


class RasterLayer:
    def __init__(self):
        return None


class VectorLayer:
    def __init_(self):
        return None


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
        return None

    def addRasterLayerAsOverlay(self,rasterLayer):
        # http://qingkaikong.blogspot.in/2016/06/using-folium-5-image-overlay-overlay.html
        # http://nbviewer.jupyter.org/github/python-visualization/folium/blob/master/examples/ImageOverlay.ipynb
        return 0

    def addVectorLayerAsOverlay(self,vectorLayer):
        # https://ocefpaf.github.io/python4oceanographers/blog/2015/12/14/geopandas_folium/
        # http://andrewgaidus.com/leaflet_webmaps_python/
        return 0

"""
Manages all test functions for SpatialIO
"""
def test():
    doctest.testmod()
