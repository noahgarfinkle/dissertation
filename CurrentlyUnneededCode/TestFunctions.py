import matplotlib.pyplot as plt
%matplotlib inline
plt.plot(range(0,1000))
import rasterio as rs
import geopandas as pd
import numpy as np
#testRasterPath = "/home/noah/Missori_FLW/DataHUB/R_Friendly/Raster/DEM_CMB_ELV_SRTMVF2.tif"
testRasterPath = "/home/noah/GIT/dissertation/test_data/testelevunproj.tif"
raster = rs.open(testRasterPath)
rasterData = raster.read()
plt.imshow(rasterData[0,:,:])
rasterData.shape
print(raster.crs)
print(raster.transform)
print(raster.width,raster.height)
print(raster.count)
print(raster.indexes)
raster.window(-95.99999999999999, 0.00027777777777777924, 41.00000190734863, -0.0002777777777777771)


import georasters as gr
graster = gr.from_file(testRasterPath)
graster.plot()


graster.mean()
graster.sum()
graster.std()
#df = graster.to_geopandas()
graster2 = graster **2
graster2.to_tiff('./tmp/testexponential')

value = graster.map_pixel_location(100,100)
#graster.map_pixel(value[1],value[0])

from osgeo import gdal, ogr, osr
gtif = gdal.Open(testRasterPath)
print gtif.GetMetadata()
gtif.RasterCount
for band in range(gtif.RasterCount):
    band += 1
    srcband = gtif.GetRasterBand(band)
    if srcband is None:
        continue
    stats = srcband.GetStatistics(True,True)
    if stats is None:
        continue
    print "[ STATS ] =  Minimum=%.3f, Maximum=%.3f, Mean=%.3f, StdDev=%.3f" % ( \
                stats[0], stats[1], stats[2], stats[3] )
stats

srcband.GetScale()
srcband.GetMinimum()
srcband.GetMaximum()
srcband.GetUnitType()
srcband.GetColorInterpretation()
srcband.GetColorTable()
srcband.SetUnitType('m')

from enum import Enum
class CRS(Enum):
    WGS84 = 4326
    WMAS = 3857


class Raster:
    def __init__(self):
        self.rasterPath = None
        self.raster = None
        self.crs = "None"

    def from_empty(self,lx,ly,ux,uy,crs,scale):
        return 0

    def from_file(self,raster_path):
        self.rasterPath = raster_path
        self.raster = gdal.Open(raster_path)
        proj = self.raster.GetProjection()
        srs = osr.SpatialReference(wkt=proj)
        #epsg = srs.GetAttrValue('AUTHORITY',1) # https://gis.stackexchange.com/questions/267321/extracting-epsg-from-a-raster-using-gdal-bindings-in-python
        self.crs = srs.ExportToWkt()

    def plot(self):
        return 0

    def export(self,newPath):
        self.rasterPath = newPath

    def reproject(self,crs=CRS.WMAS):rl.minimum

        tmpRaster = "./tmp/tmp.tif"
        spatRef = osr.SpatialReference()
        spatRef.ImportFromEPSG(crs.value)
        gdal.Warp(tmpRaster,self.raster,dstSRS=spatRef)
        self.raster = gdal.Open(tmpRaster)
        self.crs = spatRef.ExportToWkt()
        self.rasterPath = "In memory: export to update"


r = Raster()
r.from_file(testRasterPath)

r.crs
r.reproject()
r.crs

# test using doctest, https://docs.python.org/2/library/doctest.html
"""
This is a test
>>> factorial(5)

class Raster:
    def __init__(self):
        self.rasterPath = None
        self.raster = None
        self.crs = "None"

    def from_empty(self,lx,ly,ux,uy,crs,scale):
        return 0

    def from_file(self,raster_path):
        self.rasterPath = raster_path
        self.raster = gdal.Open(raster_path)
        proj = self.raster.GetProjection()
        srs = osr.SpatialReference(wkt=proj)
        #epsg = srs.GetAttrValue('AUTHORITY',1) # https://gis.stackexchange.com/questions/267321/extracting-epsg-from-a-raster-using-gdal-bindings-in-python
        self.crs = srs.ExportToWkt()

    def plot(self):
        return 0

    def export(self,newPath):
        self.rasterPath = newPath

    def reproject(self,crs=CRS.WMAS):rl.minimum

        tmpRaster = "./tmp/tmp.tif"
        spatRef = osr.SpatialReference()
        spatRef.ImportFromEPSG(crs.value)
        gdal.Warp(tmpRaster,self.raster,dstSRS=spatRef)
        self.raster = gdal.Open(tmpRaster)
        self.crs = spatRef.ExportToWkt()
        self.rasterPath = "In memory: export to update"


r = Raster()
r.from_file(testRasterPath)

r.crs
r.reproject()
r.crs
120
"""
def factorial(n):
    """Return the factorial of n, an exact integer >= 0.
    If the result is small enough to fit in an int, return an int.
    Else return a long.

    >>> [factorial(n) for n in range(6)]
    [1, 1, 2, 6, 24, 120]
    >>> [factorial(long(n)) for n in range(6)]
    [1, 1, 2, 6, 24, 120]
    >>> factorial(30)
    265252859812191058636308480000000L
    >>> factorial(30L)
    265252859812191058636308480000000L
    >>> factorial(-1)
    Traceback (most recent call last):
        ...
    ValueError: n must be >= 0

    Factorials of floats are OK, but the float must be an exact integer:
    >>> factorial(30.1)
    Traceback (most recent call last):
        ...
    ValueError: n must be exact integer
    >>> factorial(30.0)
    265252859812191058636308480000000L

    It must also not be ridiculously large:
    >>> factorial(1e100)
    Traceback (most recent call last):
        ...
    OverflowError: n too large
    """

    import math
    if not n >= 0:
        raise ValueError("n must be >= 0")
    if math.floor(n) != n:
        raise ValueError("n must be exact integer")
    if n+1 == n:  # catch a value like 1e300
        raise OverflowError("n too large")
    result = 1
    factor = 2
    while factor <= n:
        result *= factor
        factor += 1
    return result

import doctest
doctest.testmod()
