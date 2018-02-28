# -*- coding: utf-8 -*-
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

import SpatialIO as io

## HELPFUL FOR DEBUGGING
# %matplotlib inline
# pd.options.display.max_columns = 300

## SETUP
validStats = utils.VALID_STATS

""" REFERENCES
http://lxml.de/tutorial.html
https://mapbox.s3.amazonaws.com/playground/perrygeo/rasterio-docs/cookbook.html#rasterizing-geojson-features
http://skipperkongen.dk/2012/03/06/hello-world-of-raster-creation-with-gdal-and-python/
https://github.com/mapplus/qgis-scripts/blob/master/scripts/Raster%20Euclidean%20Distance%20Analysis.py
https://stackoverflow.com/questions/30740046/calculate-distance-to-nearest-feature-with-geopandas
http://portolan.leaffan.net/creating-sample-points-with-ogr-and-shapely-pt-2-regular-grid-sampling/
https://gis.stackexchange.com/questions/159624/converting-large-rasters-to-numpy
https://gis.stackexchange.com/questions/264793/crop-raster-in-memory-with-python-gdal-bindings
http://pythonhosted.org/rasterstats/manual.html#zonal-statistics -> can define custom statistics!!!
https://github.com/perrygeo/python-rasterstats/blob/master/src/rasterstats/main.py
https://github.com/perrygeo/python-rasterstats/blob/master/src/rasterstats/utils.py
https://github.com/SALib/SALib -> Sensitivity analysis software
"""

""" REFERENCE CODE
setting up blank raster: dst_ds.SetGeoTransform([topLeftX,pixel_width,0,topLeftY,0,-pixel_height])
"""

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

## FUNCTIONS
def filterDataFrameByBounds(df,lx,ly,ux,uy,bufferDistance=0):
    """ Subsets a GeoPandas DataFrame by bounding box

    Very useful for shrinking the number of features.  Currently assumes that the bounding
    box and optional buffer distance is in the same projection as the dataframe.

    Args:
        df (GeoPandas DataFrame): The dataframe to be filtered
        lx (float): Lower left x coordinate of the bounding box
        ly (float): Lower left y coordinate of the bounding box
        ux (float): Upper right x coordinate of the bounding box
        uy (float): Upper right y coordinate of the bounding box
        bufferDistance (float): Optional distance for buffering the bounding box

    Returns:
        filteredDF (GeoPandas DataFrame): A filtered copy of the original df

    Raises:
        None

    Tests:
        None
    """
    filteredDF = df.cx[lx-bufferDistance:ux+bufferDistance,ly-bufferDistance:uy+bufferDistance]
    return filteredDF

def filterDataFrameByValue(df,column,argument):
    """ Returns a subset of a GeoPandas GeoDataframe

    Currently only works for single instances of categorical variables.  For more
    complicated cases, either code directly or update this function

    Args:
        df (GeoPandas DataFrame): The dataframe to be filtered
        column (str): The string name of the dataframe column to be filtered
        argument(var): The value determining which rows to return

    Returns:
        filteredDF (GeoPandas DataFrame): A filtered copy of the original df

    Raises:
        None

    Tests:
        None
    """
    filteredDF = df[df[column]==argument]
    return filteredDF

def minimumDistanceFromPointToDataFrameFeatures(x,y,crs,df):
    """ Returns the minimum euclidean distance from a point to the dataframe

    Currently CRS does not project the point

    Args:
        x (float): x-coordinate in the same projection as the dataframe
        y (float): y-coordinate in the same projection as the dataframe
        crs (ENUM CRS): The projection of the point
        df (GeoPandas DataFrame): A dataframe of vector features

    Returns:
        minDistance (float): the smallest euclidean distance calculated

    Raises:
        None

    Tests:
        None
    """
    point = Point(x,y)
    minDistance = df.distance(point).min()
    return minDistance

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
    df_to_project.to_crs({'init':'EPSG:' + str(to_epsg)})
    reprojectedWKT = df_to_project.geometry[0].to_wkt()
    return reprojectedWKT


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

def generateEvaluationGridDataFrame(polygon,gridSpacing):
    """ Produces a GeoPandas GeoDataFrame of squares within an AOI polygon

    This produces the fundamental solution data structure of First Pass.  Should
    implement an option isSquare or isPoint to allow for faster production of both
    rasters and square Areas of Interest.  Additionally, should implement automatic
    optimization of gridSpacing.

    Args:
        polygon (Shapely Polygon): A representation of the area of interest
        gridSpacing (float): The size of the area to discretize

    Returns:
        squareList (GeoPandas GeoDataFrame): Each row represents a candidate site
            for evaluation

    Raises:
        None

    Tests:
        None
    """
    squareList = []
    bounds = polygon.bounds
    ll = bounds[:2]
    ur = bounds[2:]
    # https://stackoverflow.com/questions/30457089/how-to-create-a-polygon-given-its-point-vertices
    start = datetime.datetime.now()
    for x in floatrange(ll[0],ur[0],gridSpacing):
        for y in floatrange(ll[1],ur[1],gridSpacing):
            square = Polygon([[x,y],[x+gridSpacing,y],[x+gridSpacing,y+gridSpacing],[x,y+gridSpacing]])
            if square.within(polygon):
                squareList.append(square)
    end = datetime.datetime.now()
    timeElapsed = end - start
    nFeatures = len(squareList)
    print "Generated %s squares in %s seconds" %(nFeatures,timeElapsed.seconds)
    evaluationDF = gpd.GeoDataFrame(squareList)
    evaluationDF.columns = ['geometry']
    return evaluationDF

def airfieldBuilder(aoiPolygon, units="m", length="1981", width="97.50", gridSpacing="800",
                    rotationUnits="degrees", rotationStart="0", rotationStop="180",
                    rotationSpacing="90"):
    """ Produces a GeoPandas GeoDataFrame of airfields similar to the TASS methodology

    This produces the fundamental solution data structure of First Pass, a
    GeoPandas GeoDataFrame sized and rotated as specified, and falling exclusively
    within the aoiPolygon, ideally which has already had no-build areas removed.

    Args:
        aoiPolygon (Shapely Polygon): A representation of the area of interest,
            ideally with holes removed for areas which should not fall within
            candidate solutions
        units (str): The distance units to utilize, typically 'm' for meters
        length (float): Length of the airfield, passed as a string from XML but
            converted to float.  Includes any buffer areas.
        width (float): Width of the airfield, passed as a string from XML but
            converted to float.  Includes any buffer areas.
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
        airfieldList (GeoPandas GeoDataFrame): Each row represents a candidate site
            for evaluation

    Raises:
        None

    Todo:
        * Implement units and rotationUnits

    Tests:
        None
    """
    try:
        # Ensure correct types, converting units of measure to float
        length=float(length)
        width=float(width)
        gridSpacing=float(gridSpacing)
        rotationStart=float(rotationStart)
        rotationStop=float(rotationStop)
        rotationSpacing=float(rotationSpacing)

    except Exception, e:
        pass

    # enforce conversions to correct types
    return None

def polygonBuilder(aoiPolygon, epsg="3857", wkt="POLYGON ((0 0, 400 0, 400 800, 0 800, 0 0))",
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

def filterByVectorBufferDistance(dfToFilter,vectorFilePath,bufferDistance,removeIntersected=True):
    """ Utilizes shapely hack to include/exclude buffers faster than Euclidean distance

    This produces the fundamental solution data structure of First Pass, a
    GeoPandas GeoDataFrame sized and rotated as specified, and falling exclusively
    within the aoiPolygon, ideally which has already had no-build areas removed.
    Can take arbitrary polygon definitions in WKT.

    Args:
        dfToFilter (GeoPandas GeoDataFrame): The dataframe to be filtered
        vectorFilePath (str): Path to a vector geometry
        bufferDistance (float): Distance to buffer vectorDF
        removeIntersected (Bool): If True, removes any rows in dfToFilter
            which intersect vectorDF.  If False, removes any rows which do not.

    Returns:
        filteredDF (GeoPandas GeoDataFrame): A subset of dfToFilter

    Raises:
        None

    Todo:
        * Implement units and rotationUnits

    Tests:
        None
    """
    start = datetime.datetime.now()
    vectorDF = gpd.read_file(vectorFilePath)
    returnText = "Retained"
    if removeIntersected:
        filteredDF = dfToFilter[~dfToFilter.intersects(vectorDF.buffer(bufferDistance).unary_union)]
    else:
        returnText = "Retained"
        filteredDF = dfToFilter[dfToFilter.intersects(vectorDF.buffer(bufferDistance).unary_union)]
    end = datetime.datetime.now()
    timeElapsed = end - start
    initialFeatures = len(dfToFilter.index)
    filteredFeatures = len(filteredDF.index)
    print "%s %s of %s candidates in %s seconds" %(returnText,filteredFeatures,initialFeatures,timeElapsed.seconds)
    return filteredDF

def convertRasterToNumpyArray(raster_path):
    """ Generates a numpy array from a GeoTiff


    Args:
        raster_path (str): Filepath of a GeoTiff

    Returns:
        data (Numpy Array): The first band of the raster, as an array

    Raises:
        None

    Tests:
        >>> raster_path = "./test_data/testelevunproj.tif"
        >>> convertRasterToNumpyArray(raster_path)
    """


    dataset = gdal.Open(raster_path)
    band = dataset.GetRasterBand(1)
    geotransform = dataset.GetGeoTransform()
    xinit = geotransform[0]
    yinit = geotransform[3]

    xsize = geotransform[1]
    ysize = geotransform[5]
    data = band.ReadAsArray() # whole band
    # data = band.ReadAsArray(col1, row1, col2 - col1 + 1, row2 - row1 + 1) # partial band
    print data.shape
    print np.mean(data)
    return data

def generateRasterStatisticsForDataFrame(df,raster_path,stats="count majority minority unique mean",
                                            colName="slope",isCategorical=False):
    """ Produces neighborhood statistics for a raster based on each feature in a
        GeoPandas GeoDataFrame

    Works for both categorical and continuous rasters.  Note that some stats, such
    as count, substantially increase processing time.

    Args:
        df (GeoPandas GeoDataFrame): Each row represents a polygon geometry to
            generate neighborhood statistics for
        raster_path (str): Filepath of a GeoTiff raster
        stats (str): space-separated list of valid statistics from validStats
        colName (str): What to prepend to the statistic column in the new dataframe,
            done in order to avoid repeated column names
        isCategorical (bool): Raster values will be interpreted as categorical if
            True, continuous if False.  Defaults False.

    Returns:
        newDF (GeoPandas GeoDataFrame): The input df, merged with the appropriate
            raster statistics

    Raises:
        None

    Tests:
        None
    """
    start = datetime.datetime.now()
    row_stats_df = gpd.GeoDataFrame(raster_stats(vectors=df['geometry'],raster=raster_path,stats=stats, copy_properties=True, nodata_value=0, categorical=isCategorical))
    # rename the columns
    for stat in stats.split(' '):
        newColName = "%s_%s" %(colName,stat)
        row_stats_df.rename(columns={stat:newColName}, inplace=True)
    # rename any remaining columns, such as those created using count
    # this can be accomplished because the only columsn should be geometry or colName_preceeded
    for columnName in row_stats_df.columns:
        originalName = columnName
        columnName = str(columnName)
        if columnName == "geometry" or colName in columnName:
            pass
        else:
            newColName = "%s_%s" %(colName,columnName)
            row_stats_df.rename(columns={originalName:newColName}, inplace=True)
    newDF = gpd.GeoDataFrame(pd.concat([df,row_stats_df],axis=1))
    end = datetime.datetime.now()
    timeElapsed = end - start
    processedFeatures = len(df.index)
    print "Processed %s candidates in %s seconds" %(processedFeatures,timeElapsed.seconds)
    return newDF

def queryRasterValueForPoint(x,y,raster_path,pointCRS=None,rasterCRS=None):
    """ Returns the value of a GeoTiff at a specified point

    pointCRS and rasterCRS are not yet implemented, and therefore both should
    be in the same coordinate system

    Args:
        x (float): The x-coordinate, in the same CRS as the raster
        y (float): The y-coordinate, in the same CRS as the raster
        raster_path (str): Filepath to a GeoTiff
        pointCRS (ENUM CRS): Not implemented
        rasterCRS (ENUM CRS): Not implemented

    Returns:
        pointQuery (float): The value of the raster at location (x,y)

    Raises:
        None

    Todo:
        * Implement projections

    Tests:
        None
    """
    point = "POINT(%s %s)" %(x,y)
    pointQuery = point_query(point,raster_path)
    return pointQuery

def rasterStatCroppedRaster(df,raster_path):
    """ Produces neighborhood statistics for a raster based on each feature in a
        GeoPandas GeoDataFrame, also returning a column with the raster values for
        each row

    This implementation of zonal statistics is very useful for functions such as
    cut-fill, returning numpy masked arrays of each subraster inside the
    GeoDataFrame.  Note this function currently uses the default values for
    zonal_stats, and should in the future be merged with rasterStat, with a flag
    value parameter for returning the sub-rasters

    Args:
        df (GeoPandas GeoDataFrame): Each row represents a polygon geometry to
            generate neighborhood statistics for
        raster_path (str): Filepath of a GeoTiff raster

    Returns:
        newDF (GeoPandas GeoDataFrame): A data frame with each sub-raster contained
        in the column 'mini_raster_array'

    Raises:
        None

    Tests:
        None
    """
    rasterSource = zonal_stats(df['geometry'],raster_path,all_touched=True,raster_out=True)
    rasterDF = pd.DataFrame(rasterSource)
    return rasterDF

def calculateCutFill(df,dem_path,finalElevation='mean',rasterResolution=10):
    """ Generates cut fill values based on a raster

    Need to verify that this function calculates correctly. Additionally, should
    determine raster resolution on it's own, rather than an input.  In the future,
    should return cut and fill values separately.

    Args:
        df (GeoPandas GeoDataFrame): Each row represents a polygon geometry to
            generate cut/fill values for
        raster_path (str): Filepath of a GeoTiff raster containing a digital
        elevation model in the same coordinate system as the desired results
        finalElevation (str/float): must either be number or string from validStats,
            this represents the target elevation to determine cut/fill values for
            at each pixel
        rasterResolution (float): Cut/fill volume is determined by a height change
            multiplied by the square of this number, representing the pixel cell
            size in target units of the DEM

    Returns:
        appendedDF (GeoPandas GeoDataFrame): A data frame with each sub-raster contained
        in the column 'mini_raster_array', along with masked arrays of elevation change
        in 'elevationChangeArray' and the total cut fill volume in totalCutFillVolume

    Raises:
        None

    Tests:Summary line
        None
            if 'old_geo' in df.columns:
                df = df.drop('old_geo',axis=1)

            if 'old_geometry' in df.columns:
                df = df.drop('old_geometry',axis=1)


                    #for i in range(0,len(appendedDF.index+1)):
                        #row = appendedDF[i:i+1]
    """
    # KLUDGE to remove raster stats with the same name
    if finalElevation in df.columns:
        df = df.drop(finalElevation,axis=1)
    df = df.reset_index()
    croppedRasterDF = rasterStatCroppedRaster(df,dem_path)
    appendedDF = gpd.GeoDataFrame(pd.concat([df,croppedRasterDF],axis=1))
    elevationChangeArrays = []
    totalRequiredHeightChanges = []
    totalCutFillVolumes = []
    for i,row in appendedDF.iterrows():
        #maskedRaster = row['mini_raster_array']
        maskedRaster = croppedRasterDF['mini_raster_array'][i] # KLUDGE
        maskedRaster_Array = ma.masked_array(maskedRaster)
        targetElevation = -999
        if isinstance(finalElevation,basestring):
            targetElevation = row[finalElevation]
        else:
            targetElevation = finalElevation
        requiredHeightChange = np.subtract(maskedRaster_Array,targetElevation)
        totalRequiredHeightChange = np.sum(np.abs(requiredHeightChange))
        totalCutFillVolume = totalRequiredHeightChange * rasterResolution * rasterResolution
        elevationChangeArrays.append(requiredHeightChange)
        totalCutFillVolumes.append(totalCutFillVolume)

    appendedDF['elevationChangeArray'] = elevationChangeArrays
    appendedDF['totalCutFillVolume'] = totalCutFillVolumes
    return appendedDF


def minimumDistanceFromEvaluationToDataFrameFeatures(evaluationDF,vectorDF):
        """ Implements Euclidean distance from a data frame of candiate polygons
        to a vector data frame

        Assumes that evaluationDF contains polygons and vectorDF contains vectors

        Args:
            evaluationDF (GeoPandas GeoDataFrame): Each row represents a polygon
                geometry to evaluate Euclidean distances for
            vectorDF (GeoPandas GeoDataFrame): Each row represents a vector
                geometry

        Returns:
            evaluationDF (GeoPandas GeoDataFrame): Appends the minimum distance
                to vectorDF in the column 'distance'

        Raises:
            None

        Tests:Summary line
            None
        """
        minDistances = []
        for i,row in evaluationDF.iterrows():
            minDistance = vectorDF.distance(row.geometry).min()
            minDistances.append(minDistance)
        evaluationDF['distance'] = minDistances
        return evaluationDF

def convertSubsettedEvaluationDFIntoPolygonGrid(evaluationDF, squareDimension):
    polygonEvaluations = []
    oldPolygons = []
    for geometry in evaluationDF.geometry:
        oldPolygons.append(geometry)
        centroid_x = geometry.centroid.x
        centroid_y = geometry.centroid.y
        offset = squareDimension / 2
        square = Polygon([[centroid_x - offset,centroid_y - offset],
                          [centroid_x + offset,centroid_y - offset],
                          [centroid_x + offset,centroid_y + offset],
                          [centroid_x - offset,centroid_y + offset]])
        polygonEvaluations.append(square)
    evaluationDF = evaluationDF.drop('geometry',axis=1)
    evaluationDF['geometry'] = polygonEvaluations
    evaluationDF['old_geo'] = oldPolygons
    return evaluationDF

def generateRandomLatLonPair(latMin,latMax,lonMin,lonMax):
    """ Creates random points to help test other functions

    This code is a helper function

    Args:
        latMin (float): Lower bound of latitude
        latMax (float): Upper bound of latitude
        lonMin (float): Lower bound of longitude
        lonMax (float): Upper bound of longitude

    Returns:
        lat (float): A random latitude within the specified range
        lon (float): A random longitude within the specified range

    Raises:
        None

    Tests:
        None
    """
    lat = np.random.uniform(latMin,latMax)
    lon = np.random.uniform(lonMin,lonMax)
    return lat,lon

def generateRandomCandidateDataFrame(nCandidates,latMin,latMax,lonMin,lonMax):
        """ Creates random solutionDF to help test other functions

        This code is a helper function

        Args:
            nCandidates (int): Number of candidate sites to create
            latMin (float): Lower bound of latitude
            latMax (float): Upper bound of latitude
            lonMin (float): Lower bound of longitude
            lonMax (float): Upper bound of longitude

        Returns:
            lat (float): A random latitude within the specified range
            lon (float): A random longitude within the specified range

        Raises:
            None

        Tests:
            None
        """
        lats = []
        lons = []
        scores = []
        geoms = []
        for i in range(0,nCandidates):
            lat,lon = generateRandomLatLonPair(latMin,latMax,lonMin,lonMax)
            lats.append(lat)
            lons.append(lon)
            score = np.random.randint(0,101)
            scores.append(score)
            point = Point([lon,lat])
            square = Polygon([[x,y],[x+gridSpacing,y],[x+gridSpacing,y+gridSpacing],[x,y+gridSpacing]])
            if square.within(polygon):
                squareList.append(square)
        candidateDF = gpd.GeoDataFrame({""})
        return 0


## CURRENT TEST
# First pass implementation for class
"""
aoiDF = gpd.read_file("../FLW_Missouri Mission Folder/SUPPORT/Staging.shp")
aoiDF = aoiDF.to_crs({'init':'epsg:3857'})
squareDimension = 400
aoiDF.plot()

# Airfield Objective
airfieldAOI = aoiDF[aoiDF['Stage']=='Gold'].reset_index().geometry[0]
airfieldEvaluationDataFrame = generateEvaluationGridDataFrame(airfieldAOI,100)
slopePath = '../FLW_Missouri Mission Folder/RASTER/slope_proj.tif'
airfieldSlopeEvaluationDataFrame = generateRasterStatisticsForDataFrame(airfieldEvaluationDataFrame,slopePath,stats="mean max",isCategorical=False)
airfieldSlopeEvaluationDataFrame.head()
airfieldSlopeEvaluationDataFrame.plot(column='max')
plt.hist(airfieldSlopeEvaluationDataFrame['max'])
airfieldSlopeEvaluationDataFrameSubset = airfieldSlopeEvaluationDataFrame[airfieldSlopeEvaluationDataFrame['max'] < 2]
airfieldSlopeEvaluationDataFrameSubset.head()

len(airfieldSlopeEvaluationDataFrameSubset.index)
airfieldSlopeEvaluationDataFrameSubset.plot(column='max')


largerAirfields = convertSubsettedEvaluationDFIntoPolygonGrid(airfieldSlopeEvaluationDataFrameSubset, 800)
largerAirfields.head()
largerAirfields.plot()

elevationPath = "../FLW_Missouri Mission Folder/RASTER/DEM_CMB_ELV_SRTMVF2_proj.tif"
largerAirfields.head()
airfieldEvaluationDataFrame.head()

cutFillDF = calculateCutFill(largerAirfields,elevationPath,finalElevation='mean',rasterResolution=30)
cutFillDF.head()
cutFillDF.plot(column='totalCutFillVolume')

plt.hist(cutFillDF['totalCutFillVolume'])
"""
"""
# Base Objective 1
baseObjective1AOI = aoiDF[aoiDF['Stage']=='Gold'].reset_index().geometry[0]
baseObjective1EvaluationDataFrame = generateEvaluationGridDataFrame(baseObjective1AOI,100)


# Base Objective 2
baseObjective2AOI = aoiDF[aoiDF['Stage']=='Gold'].reset_index().geometry[0]
baseObjective2EvaluationDataFrame = generateEvaluationGridDataFrame(baseObjective2AOI,100)
"""






"""
raster_path = "../FLW_Missouri Mission Folder/RASTER/DEM_CMB_ELV_SRTMVF2_proj.tif"
result_DF = generateRasterStatisticsForDataFrame(evaluationGridDataFrame,raster_path,stats="mean",isCategorical=False)
plt.figure()
plt.suptitle("Plot of mean elevation for AOI")
result_DF.plot(column='mean')
plt.savefig("./results/meanelevationforAOI.png")
evaluationGridDataFrame.plot()

start = datetime.datetime.now()
cutFillDF = calculateCutFill(evaluationGridDataFrame,raster_path,finalElevation='mean',rasterResolution=30)
end = datetime.datetime.now()
elapsedTime = end - start
nItems = len(cutFillDF.index)
print "Evaluated %s features in %s seconds" %(nItems,elapsedTime.seconds)
plt.figure()
plt.suptitle("Cut fill for evaluation cells (m3) for AOI")
array = cutFillDF['mini_raster_array'][0]
array
plt.imshow(array)
cutFillDF.head()
cutFillDF.plot(column="totalCutFillVolume")
plt.savefig("./results/cutfillquantity.png")

cutFillDF.head()

plt.figure()
plt.suptitle("Cut fill for evaluation cells (m3) for AOI")
plt.xlabel("Total volume cut/fill required (m3)")
plt.ylabel("Number of evaluation sites")
plt.hist(cutFillDF['totalCutFillVolume'])
plt.savefig("./results/cutfillhistogram.png")



meaninglessCategoricalDF = generateRasterStatisticsForDataFrame(df_subset,raster_path,isCategorical=True)
meaninglessCategoricalDF

# minimum distance from each raster cell to roads


vector_path = "./test_data/UtilityInfrastructureCrv_3.shp"
roadsDF = gpd.read_file(vector_path)
roadsDF.crs = {'init':'epsg:3857'}
roadsDF = roadsDF.to_crs({'init':'epsg:3857'})
evaluationDF = evaluationGridDataFrame
vectorDF = roadsDF
lx,ly,ux,uy = evaluationDF.total_bounds

vectorDF_filtered = filterDataFrameByBounds(vectorDF,lx,ly,ux,uy)
vectorDF_filtered.plot()

minDistances = []
for i,row in evaluationDF.iterrows():
    minDistance = vectorDF_filtered.distance(row.geometry).min()
    minDistances.append(minDistance)
evaluationDF['distance'] = minDistances
plt.figure()
f,ax = plt.subplots(1)
f.suptitle("Minimum distance from infrastructure")
evaluationDF.plot(column='distance',ax=vectorDF_filtered.plot())
plt.savefig("./results/distancefrominfrastructure.png")



a = evaluationDF[0:1]
a
a.to_html()
"""




## TESTS
# FIRST PASS IMPLEMENTATION
def testFirstPassImplementation():
    xmlPath = "./input.xml"
    raster_path = "../FLW_Missouri Mission Folder/RASTER/DEM_CMB_ELV_SRTMVF2.tif"
    vector_path = "./test_data/UtilityInfrastructureCrv_3.shp"
    df = gpd.read_file('./test_data/MO_2016_TIGER_Counties_shp/MO_2016_TIGER_Counties_shp.shp')
    df.CRS = {'init':'epsg:4326'}
    df_proj = df.to_crs({'init':'epsg:3857'})
    aoiWKT = df_proj[df_proj["NAME"] == "Pulaski"].geometry.values[0].to_wkt()
    aoiLoaded = loads(aoiWKT)

    roadsDF = gpd.read_file(vector_path)
    roadsDF.crs = {'init':'epsg:3857'}
    roadsDF = roadsDF.to_crs({'init':'epsg:3857'})

    lx,ly,ux,uy = df_proj['geometry'][4].envelope.bounds
    filteredRoads = filterDataFrameByBounds(roadsDF,lx,ly,ux,uy)
    filteredCounties = filterDataFrameByBounds(df_proj,lx,ly,ux,uy)
    filteredCounties.plot(ax=filteredRoads.plot(facecolor='red'),facecolor='Green')

def testPointDistance():
    point = Point(ux,uy)
    print roadsDF.distance(point).min()

def createTestGrid():
    aoiLoaded = loads(aoiWKT)
    testList = generateEvaluationGridDataFrame(aoiLoaded,3200)
    testList.plot()

# test.zonal_stats
def test_zonal_stats():
    raster_path = "./test_data/testelevunproj.tif"
    dfStatsCategorical = generateRasterStatisticsForDataFrame(df,raster_path,isCategorical=True)
    dfStatsCategorical.head()
    dfStatsCategorical.plot(column="mean")

    dfStatsNonCategorical = generateRasterStatisticsForDataFrame(df,raster_path,isCategorical=False)
    dfStatsNonCategorical.head()
    dfStatsNonCategorical.plot(column='majority')

# raster tests
def test_rasterStatCroppedRaster(index = 40):
    df_subset = df[index:index + 1]
    rasterDF = rasterStatCroppedRaster(df_subset,raster_path)
    masked_array = rasterDF['mini_raster_array'][0]
    masked_array_np_masked = ma.masked_array(masked_array)
    masked_array_np = np.array(masked_array)
    plt.figure()
    plt.imshow(masked_array_np_masked)
    plt.figure()
    plt.imshow(masked_array_np)
    print "Mean Unmasked: %s, Mean Masked: %s" %(np.mean(masked_array_np),np.mean(masked_array_np_masked))
    np.mean(masked_array_np)
    np.mean(masked_array_np_masked)
    return df['geometry'][index]



# cut fill tests
def testCutFill():
    index = 40
    df_subset = df[index:index + 2]
    df_subset = df_subset.reset_index()
    dem_path = raster_path
    finalElevation = 'mean'

    appendedDF = calculateCutFill(df_subset,raster_path)
    appendedDF
    appendedDF.plot()
    plt.imshow(appendedDF['elevationChangeArray'][1])
