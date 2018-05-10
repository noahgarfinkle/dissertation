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
import time

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

def wktToShapelyPolygon(wkt,epsg,to_epsg=None):
    """ Creates a Shapely polygon from wkt, and projects if specified

    Args:
        wkt (str): A string of well-known-text
        epsg (int): The current projection of wkt
        to_epsg (int): The desired projection of wkt, not used if None

    Returns:
        wktPolygon (Shapely Geometry): Shapely representation of wkt

    Raises:
        None

    Tests:
        None
    """
    wktPolygon = loads(wkt)
    if to_epsg:
        df_to_project = gpd.GeoDataFrame([wktPolygon])
        df_to_project.columns = ['geometry']
        df_to_project.crs = {'init':'EPSG:' + str(epsg)}
        df_to_project = df_to_project.to_crs({'init':'EPSG:' + str(to_epsg)})
        wktPolygon = df_to_project.geometry[0]
    return wktPolygon

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
    row_stats_df.index = df.index

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

    #appendedDF['elevationChangeArray'] = elevationChangeArrays
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

## ENSITE FUNCTIONS
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

def buildCategoricalRasterStatFromXML(evaluationDF,criteriaRow):
    """ Converts XML into categorical raster statistic evaluation

        Evaluation function

    Args:
        evaluationDF (lxml): Set of candidate solutions
        criteriaRow (lxml): CategoricalRasterStat

    Returns:
        evaluationDF (GeoPandas GeoDataFrame): Scored and subsetted dataframe
            based upon analysis

    Raises:
        None

    Tests:
        None
    """
    print "Categorical Raster Stat: %s.  Evaluating %s candidates." %(criteriaRow.attrib['criteriaName'],len(evaluationDF.index))
    criteriaName = criteriaRow.attrib['criteriaName']
    layerPath = criteriaRow.attrib['layerPath']
    lowerBound = str(criteriaRow.attrib['lowerBound'])
    upperBound = str(criteriaRow.attrib['upperBound'])
    start = datetime.datetime.now()


    if lowerBound == "-INF":
        lowerBound = -1.0
    else:
        lowerBound = float(lowerBound)
    if upperBound == "INF":
        upperBound = 101.0
    else:
        upperBound = float(upperBound)

    valueList = criteriaRow.attrib['valueList']


    initialDataFrameSize = len(evaluationDF.index)

    evaluationDF = generateRasterStatisticsForDataFrame(evaluationDF,layerPath,stats="count",colName=criteriaName,isCategorical=True)
    end_EvaluationDF = datetime.datetime.now()
    timeElapsed = end_EvaluationDF - start
    print "generateRasterStatisticsForDataFrame took %s seconds" %(timeElapsed.seconds)
    # replace NA values with zero, note this may need to be moved into the first pass function to make sure I do not unintentionally overwrite other data
    evaluationDF = evaluationDF.fillna(0)
    end_fillna = datetime.datetime.now()
    timeElapsed = end_fillna - end_EvaluationDF
    print "fillna took %s seconds" %(timeElapsed.seconds)
    # calculate percentages
    values = valueList.split(',')
    totalCountColumnName = "%s_count" %(criteriaName)
    countColumnNames = []
    for value in values:
        countColumnName = "%s_%s" %(criteriaName,value)
        countColumnNames.append(countColumnName)

    evaluationDF[criteriaName] = 0
    for countColumnName in countColumnNames:
        if evaluationDF.columns.contains(countColumnName):
            evaluationDF[criteriaName] += evaluationDF[countColumnName]

    evaluationDF[criteriaName] = evaluationDF[criteriaName] / evaluationDF[totalCountColumnName] * 100.0
    end_CreatingCriteria = datetime.datetime.now()
    timeElapsed = end_CreatingCriteria - end_fillna
    print "creatingCriteriaName took %s seconds" %(timeElapsed.seconds)

    scores = criteriaRow.find("Scores")
    weight = scores.attrib['weight']
    isZeroExclusionary = scores.attrib['isZeroExclusionary']
    default = scores.attrib['default']
    scoreStructure = []
    for scoreRow in scores:
        lowerBoundInclusive = str(scoreRow.attrib['lowerBoundInclusive'])
        upperBoundExclusive = str(scoreRow.attrib['upperBoundExclusive'])
        score = str(scoreRow.attrib['score'])
        scoreSet = [lowerBoundInclusive,upperBoundExclusive,score]
        scoreStructure.append(scoreSet)
    evaluationDF = scoreDF(evaluationDF,criteriaName,scoreStructure,isZeroExclusionary=isZeroExclusionary)
    end_scoring = datetime.datetime.now()
    timeElapsed = end_scoring - end_CreatingCriteria
    print "scoring took %s seconds" %(timeElapsed.seconds)
    """
        # trim the dataframe
    if isZeroExclusionary == "True":
        initialDataFrameSize = len(evaluationDF.index)
        evaluationDF = evaluationDF[evaluationDF[criteriaName] > lowerBound]
        numberAfterLowerBoundFilter = len(evaluationDF.index)
        evaluationDF = evaluationDF[evaluationDF[criteriaName] < upperBound]
        numberAfterUpperBoundFilter = len(evaluationDF.index)
        """

    print "Retained %s of %s candidates" %(len(evaluationDF.index),initialDataFrameSize)

#    print "Retained %s of %s candidates, with %s removed for being too low and %s removed for being too high" %(numberAfterUpperBoundFilter,initialDataFrameSize,initialDataFrameSize-numberAfterLowerBoundFilter,numberAfterLowerBoundFilter-numberAfterUpperBoundFilter)
    return evaluationDF

def buildContinuousRasterStatFromXML(evaluationDF,criteriaRow):
    """ Converts XML into continuous raster statistic evaluation

        Evaluation function

    Args:
        evaluationDF (lxml): Set of candidate solutions
        criteriaRow (lxml): ContinuousRasterStat

    Returns:
        evaluationDF (GeoPandas GeoDataFrame): Scored and subsetted dataframe
            based upon analysis

    Raises:
        None

    Tests:
        None
    """
    print "Continuous Raster Stat: %s.  Evaluating %s candidates." %(criteriaRow.attrib['criteriaName'],len(evaluationDF.index))

    criteriaName = criteriaRow.attrib['criteriaName']
    layerPath = criteriaRow.attrib['layerPath']
    lowerBound = str(criteriaRow.attrib['lowerBound'])
    upperBound = str(criteriaRow.attrib['upperBound'])

    if lowerBound == "-INF":
        lowerBound = -1.0
    else:
        lowerBound = float(lowerBound)
    if upperBound == "INF":
        upperBound = 100000.0
    else:
        upperBound = float(upperBound)

    stat = criteriaRow.attrib['stat']
    initialDataFrameSize = len(evaluationDF.index)

    evaluationDF = generateRasterStatisticsForDataFrame(evaluationDF,layerPath,stats=stat,colName=criteriaName,isCategorical=False)
    evaluationColumnName = "%s_%s" %(criteriaName,stat)
    # KLUDGE- copy the evaluation column
    evaluationDF[criteriaName] = evaluationDF[evaluationColumnName] # in the future replace instead of copy

    scores = criteriaRow.find("Scores")
    weight = scores.attrib['weight']
    isZeroExclusionary = scores.attrib['isZeroExclusionary']
    default = scores.attrib['default']
    scoreStructure = []
    for scoreRow in scores:
        lowerBoundInclusive = str(scoreRow.attrib['lowerBoundInclusive'])
        upperBoundExclusive = str(scoreRow.attrib['upperBoundExclusive'])
        score = str(scoreRow.attrib['score'])
        scoreSet = [lowerBoundInclusive,upperBoundExclusive,score]
        scoreStructure.append(scoreSet)
    evaluationDF = scoreDF(evaluationDF,criteriaName,scoreStructure,isZeroExclusionary=isZeroExclusionary)
    print "Retained %s of %s candidates" %(len(evaluationDF.index),initialDataFrameSize)
    #print "Retained %s of %s candidates, with %s removed for being too low and %s removed for being too high" %(numberAfterUpperBoundFilter,initialDataFrameSize,initialDataFrameSize-numberAfterLowerBoundFilter,numberAfterLowerBoundFilter-numberAfterUpperBoundFilter)
    return evaluationDF

    """# trim the dataframe
    if isZeroExclusionary == "True":
        initialDataFrameSize = len(evaluationDF.index)
        evaluationColumnName = "%s_%s" %(criteriaName,stat)
        evaluationDF = evaluationDF[evaluationDF[evaluationColumnName] > lowerBound]
        numberAfterLowerBoundFilter = len(evaluationDF.index)
        evaluationDF = evaluationDF[evaluationDF[evaluationColumnName] < upperBound]
        numberAfterUpperBoundFilter = len(evaluationDF.index)
        """

def buildDistanceFromVectorLayerFromXML(evaluationDF,criteriaRow):
    """ Converts XML into vector distance evaluation

        Evaluation function

    Args:
        evaluationDF (lxml): Set of candidate solutions
        criteriaRow (lxml): filterByVectorBufferDistance

    Returns:
        evaluationDF (GeoPandas GeoDataFrame): Scored and subsetted dataframe
            based upon analysis

    Raises:
        None

    Tests:
        None
    """
    print "Distance From Vector Layer: %s.  Evaluating %s candidates." %(criteriaRow.attrib['criteriaName'],len(evaluationDF.index))
    criteriaName = criteriaRow.attrib['criteriaName']
    layerPath = criteriaRow.attrib['layerPath']
    lowerBound = str(criteriaRow.attrib['lowerBound'])
    upperBound = str(criteriaRow.attrib['upperBound'])

    if lowerBound == "-INF":
        lowerBound = -1.0
    else:
        lowerBound = float(lowerBound)
    if upperBound == "INF":
        upperBound = 100000.0
    else:
        upperBound = float(upperBound)

    evaluationDF = filterByVectorBufferDistance(evaluationDF,layerPath,lowerBound,removeIntersected=True)
    evaluationDF = filterByVectorBufferDistance(evaluationDF,layerPath,upperBound,removeIntersected=False)
    vectorQAFNameKludge = "%s_QAF" %(criteriaName)
    evaluationDF[vectorQAFNameKludge] = 100 # KLUDGE, because so expensive to actually calculate
    return evaluationDF

""" Currently does not actually provide a score for vector distance
    scores = criteriaRow.find("Scores")
    weight = scores.attrib['weight']
    isZeroExclusionary = scores.attrib['isZeroExclusionary']
    default = scores.attrib['default']
    scoreStructure = []
    for scoreRow in scores:
        lowerBoundInclusive = str(scoreRow.attrib['lowerBoundInclusive'])
        upperBoundExclusive = str(scoreRow.attrib['upperBoundExclusive'])
        score = str(scoreRow.attrib['score'])
        scoreSet = [lowerBoundInclusive,upperBoundExclusive,score]
        scoreStructure.append(scoreSet)
    evaluationDF = scoreDF(evaluationDF,"totalCutFillVolume",scoreStructure,isZeroExclusionary=False)
"""

def buildCutFillFromXML(evaluationDF,criteriaRow):
    """ Converts XML into cut/fill evaluation

        Evaluation function

    Args:
        evaluationDF (lxml): Set of candidate solutions
        criteriaRow (lxml): CutFill

    Returns:
        evaluationDF (GeoPandas GeoDataFrame): Scored and subsetted dataframe
            based upon analysis

    Raises:
        None

    Tests:
        None
    """
    print "Cut Fill: %s.  Evaluating %s candidates." %(criteriaRow.attrib['criteriaName'],len(evaluationDF.index))
    criteriaName = criteriaRow.attrib['criteriaName']
    layerPath = criteriaRow.attrib['layerPath']
    lowerBound = str(criteriaRow.attrib['lowerBound'])
    upperBound = str(criteriaRow.attrib['upperBound'])

    if lowerBound == "-INF":
        lowerBound = -1.0
    else:
        lowerBound = float(lowerBound)
    if upperBound == "INF":
        upperBound = 10000000.0
    else:
        upperBound = float(upperBound)

    evaluationDF = calculateCutFill(evaluationDF,layerPath,finalElevation='mean',rasterResolution=1)
    evaluationDF[criteriaName] = evaluationDF["totalCutFillVolume"] # KLUDGE, in the future replace the column name
    initialNumber = len(evaluationDF.index)
    #evaluationDF = evaluationDF[evaluationDF["totalCutFillVolume"] < upperBound]
    finalNumber = len(evaluationDF.index)
    scores = criteriaRow.find("Scores")
    weight = scores.attrib['weight']
    isZeroExclusionary = scores.attrib['isZeroExclusionary']
    default = scores.attrib['default']
    scoreStructure = []
    for scoreRow in scores:
        lowerBoundInclusive = str(scoreRow.attrib['lowerBoundInclusive'])
        upperBoundExclusive = str(scoreRow.attrib['upperBoundExclusive'])
        score = str(scoreRow.attrib['score'])
        scoreSet = [lowerBoundInclusive,upperBoundExclusive,score]
        scoreStructure.append(scoreSet)
    evaluationDF = scoreDF(evaluationDF,criteriaName,scoreStructure)

    print "Retained %s of %s candidates, with %s removed for cut fill being too high" %(finalNumber,initialNumber,initialNumber-finalNumber)

    return evaluationDF

def scoreDF(df,criteriaColumnName,scoreStructure,isZeroExclusionary = False):
    # score data structure is list of [loweBoundInclusive,upperBoundExclusive,score], everything outside should default to 0
    initialSize = len(df.index)
    qafName = "%s_QAF" %(criteriaColumnName)
    df[qafName] = 0 # this also sets the fallback position
    for scoreSet in scoreStructure:
        lowerBound = scoreSet[0]
        upperBound = scoreSet[1]
        score = scoreSet[2]
        if lowerBound == "-INF":
            lowerBound = -1
        if upperBound == "INF":
            upperBound = 100000000
        lowerBound = float(lowerBound)
        upperBound = float(upperBound)
        score = float(score)
        affectedRows =(df[criteriaColumnName] >= lowerBound) & (df[criteriaColumnName] <= upperBound)# upper bound should actually be exclusive
        df.loc[affectedRows,qafName] = score
        if isZeroExclusionary == "True":
            df = df[df[qafName] != 0]
        filteredSize = len(df.index)
    print "scoreDF for column %s retained %s of %s candidates" %(criteriaColumnName,filteredSize,initialSize)
    return df

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

def returnCriteriaMetadataForMCDA(criteriaRow):
    criteriaName = criteriaRow.attrib["criteriaName"]
    scores = criteriaRow.find("Scores")
    weight = scores.attrib["weight"]
    isZeroExclusionary = scores.attrib["isZeroExclusionary"]
    return criteriaName,weight,isZeroExclusionary

def runMSSPIX(xmlPath,returnDFInsteadOfLayerID=False):
    """ Runs site search for a given xml document

        Evaluation function

    Args:
        xmlPath (str): Path to input.xml

    Returns
        layerID (list<int>): The returned layer id for each evaluation in the
            database

    Raises:
        None

    Todo:
        * Currently rewritten to only write polygon layers

    Tests:
        > layerIDs = runMSSPIX("C:/Users/RDCERNWG/Documents/GIT/FLW_Missouri Mission Folder/RESULTS/Airfield 7.xml")
    """
    tree = ET.parse(xmlPath)
    root = tree.getroot()

    resultDir = root.attrib['resultDir']
    studyID = root.attrib['studyID']
    epsg = root.attrib['epsg']
    print "%s %s %s" %(resultDir,studyID,epsg)

    siteSearches = root.find("SiteSearches")
    layerIDs = []
    evaluationDFs = []
    for siteSearch in siteSearches:
        siteSearch_studyObjectiveID = siteSearch.attrib['studyObjectiveID']
        siteSearch_layerID = siteSearch.attrib['layerID']
        siteSearch_type = siteSearch.attrib['type']
        siteSearch_name = siteSearch.attrib['name']
        siteSearch_note = siteSearch.attrib['note']
        siteSearch_nReturn = siteSearch.attrib['nReturn']

        siteConfiguration = siteSearch.find("SiteConfiguration")[0]
        if siteConfiguration.tag == "WKTTemplate":
            print "WKT Template"

        searchParameters = siteSearch.find("SearchParameters")[0]
        if searchParameters.tag == "GriddedSearch":
            evaluationDF = buildGriddedSearchFromXML(siteConfiguration,searchParameters)

        siteEvaluation = siteSearch.find("SiteEvaluation")
        weights = []
        qafNames = []
        scoringDict = {}
        criteriaCount = 0
        for criteriaRow in siteEvaluation:
            criteriaCount += 1
            # set the column name if none
            if criteriaRow.attrib["criteriaName"] == "None":
                criteriaRow.attrib["criteriaName"] = "Criteria_%s" %(criteriaCount)

            # Get the metadata needed for scoring
            criteriaName,weight,isZeroExclusionary = returnCriteriaMetadataForMCDA(criteriaRow)
            qafName = "%s_QAF" %(criteriaName)
            weights.append(float(weight))
            qafNames.append(qafName)
            scoringDict[qafName] = float(weight)

            # Parse based on type of criteria
            if criteriaRow.tag == "CategoricalRasterStat":
                evaluationDF = buildCategoricalRasterStatFromXML(evaluationDF,criteriaRow)
                criteria1DF = evaluationDF
            if criteriaRow.tag == "ContinuousRasterStat":
                evaluationDF = buildContinuousRasterStatFromXML(evaluationDF,criteriaRow)
                criteria2DF = evaluationDF
            if criteriaRow.tag == "DistanceFromVectorLayer":
                evaluationDF = buildDistanceFromVectorLayerFromXML(evaluationDF,criteriaRow)
                criteria3DF = evaluationDF
            if criteriaRow.tag == "CutFill":
                evaluationDF = buildCutFillFromXML(evaluationDF,criteriaRow)
                criteria4DF = evaluationDF

        # build the weights
        totalWeight = sum(weights)
        weightedMCDAColumns = []
        for qafName in qafNames:
            scoringDict[qafName] /= totalWeight
            assignedWeight = scoringDict[qafName]
            weightedMCDAColumn = "%s_weighted" %(qafName)
            weightedMCDAColumns.append(weightedMCDAColumn)
            evaluationDF[weightedMCDAColumn] = evaluationDF[qafName] * assignedWeight

        # Build the total score
        evaluationDF["MCDA_SCORE"] = 0
        for weightedMCDAColumn in weightedMCDAColumns:
            evaluationDF["MCDA_SCORE"] += evaluationDF[weightedMCDAColumn]

        # Build the standardized score
        maxScore = max(evaluationDF["MCDA_SCORE"])
        evaluationDF["MCDA_SCORE_STANDARDIZED"] = evaluationDF["MCDA_SCORE"] / maxScore * 100.0
        evaluationDFs.append(evaluationDF)

        ensiteLayerName = "%s_%s" %(siteSearch_name,time.strftime("%Y_%m_%d_%H_%M_%S"))

        start = datetime.datetime.now()
        layerID = io.dataFrameToENSITEDatabase(evaluationDF,studyID,ensiteLayerName)
        end = datetime.datetime.now()
        timeElapsed = end - start
        print "writing to the database took %s seconds" %(timeElapsed.seconds)

        layerIDs.append(layerID)
    if returnDFInsteadOfLayerID:
        return evaluationDFs, siteEvaluation
    else:
        return layerIDs

## NEW VERSIONS



## CURRENT TEST

## TESTS
