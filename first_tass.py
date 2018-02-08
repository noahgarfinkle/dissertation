import numpy as np
import pandas as pd
from shapely import affinity
from shapely.wkt import loads
from shapely.geometry import Polygon, Point, shape
import random
import itertools
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.font_manager as font_manager
from descartes import PolygonPatch
import SpatialIO as io
import fiona
import geopandas as gpd

# https://gis.stackexchange.com/questions/113799/how-to-read-a-shapefile-in-python
shape = fiona.open('./test_data/S_HUC_Ar.shp')
feature = shape.shapeRecords()[0]
first = feature.shape.__geo_interface__
shp_geom = shape(first['geometry']

df = gpd.read_file('./test_data/S_HUC_Ar.shp')
%matplotlib inline
df.plot()

# http://portolan.leaffan.net/creating-sample-points-with-ogr-and-shapely-pt-2-regular-grid-sampling/
def floatrange(start, stop, step):
    while start < stop:
        yield start
        start += step

def generateGrid(polygon,gridSpacing):
    bounds = polygon.bounds
    ll = bounds[:2]
    ur = bounds[2:]
    for x in floatrange(ll[0],ur[0],gridSpacing):
        for y in floatrange(ll[1],ur[1],gridSpacing):
            point = Point(x,y)
            if point.within(polygon):
                print point

polygon = Polygon([(0,0), (6,0), (0,6)])
generateGrid(polygon,1.5)

def RegularGridSampler(PolygonPointSampler):
    def __init__(self, polygon = '', x_interval = 100, y_interval = 100):
        super(self.__class__, self).__init__(polygon)
        self.x_interval = x_interval
        self.y_interval = y_interval

    def perform_sampling(self):
        u"""
        Perform sampling by substituting the polygon with a regular grid of
        sample points within it. The distance between the sample points is
        given by x_interval and y_interval.
        """
        if not self.prepared:
            self.prepare_sampling()
        ll = self.polygon.bounds[:2]
        ur = self.polygon.bounds[2:]
        low_x = int(ll[0]) / self.x_interval * self.x_interval
        upp_x = int(ur[0]) / self.x_interval * self.x_interval + self.x_interval
        low_y = int(ll[1]) / self.y_interval * self.y_interval
        upp_y = int(ur[1]) / self.y_interval * self.y_interval + self.y_interval

        for x in floatrange(low_x, upp_x, self.x_interval):
            for y in floatrange(low_y, upp_y, self.y_interval):
                p = Point(x, y)
                if p.within(self.polygon):
                    self.samples.append(p)

a = RegularGridSampler(polygon=df)
df.bounds
df.bounds.min()
df.bounds.max()
df[:1].bounds
subset = df[:1]
subset.plot()
bounds = subset.bounds

ptList = []
for x in floatrange(bounds['minx'][0],bounds['maxx'][0],0.1):
    for y in floatrange(bounds['miny'][0],bounds['maxy'][0],0.1):
        point = Point(x,y)
        if subset.contains(point)[0]:
            ptList.append(point)


df2 = gpd.GeoDataFrame(ptList)
df2.columns = ['geometry']
plt.hold()

subset.plot()
df2.plot()
# http://geopandas.org/mapping.html
base = subset.plot()
df2.plot(ax=base)
