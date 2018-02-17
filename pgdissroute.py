# -*- coding: utf-8 -*-
"""
Implements routing using pgrouting
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
__date_created__ = "21 January 2018"

## IMPORTS
import psycopg2
import geopandas as gpd
import pandas as pd
import folium
import folium.plugins as plugins
from folium.plugins import MarkerCluster

## HELPFUL FOR DEBUGGING
# %matplotlib inline
# pd.options.display.max_columns = 300

## TODO
# Implement custom costs

"""REFERENCES
https://ocefpaf.github.io/python4oceanographers/blog/2015/12/14/geopandas_folium/
https://stackoverflow.com/questions/44778/how-would-you-make-a-comma-separated-string-from-a-list

"""

## SETUP
connString = "dbname='routing' user='postgres' host='localhost' password='postgres'"
conn = psycopg2.connect(connString)
cur = conn.cursor()

## CLASSES


## FUNCTIONS
def get_nearest_node(lon,lat):
    """ Finds the node in the routing table nearest to the coordinates

    Searches the routing table and returns a dataframe with the ndoe id and
    distance from the longitude and latitude, in EPSG:4326

    Args:
        lon (float): Longitude
        lat (float): Latitude

    Returns:
        nearestNode (GeoPandas GeoDataFrame): Contains a single node with the node
            id and distance to the node

    Raises:
        None

    Todo:
        * Consume a geometry or GeoDataFrame to support projections automatically,
            or consume (x,y,crs)

    Tests:
        >>> get_nearest_node(-92.1647,37.7252)
        node_id = 634267, dist = 124
    """
    sql_CreateFunction = """CREATE OR REPLACE FUNCTION get_nearest_node
        (IN x_long double precision, IN y_lat double precision) -- input parameters
        RETURNS TABLE -- structure of output
        (
          node_id bigint ,
          dist integer -- distance to the nearest station
        ) AS $$

        BEGIN

        RETURN QUERY

          SELECT id as node_id,
             CAST
             (st_distance_sphere(the_geom, st_setsrid(st_makepoint(x_long,y_lat),4326)) AS INT)
             AS d
          FROM ways_vertices_pgr
          ORDER BY the_geom <-> st_setsrid(st_makepoint(x_long, y_lat), 4326)
          LIMIT 1;

          -- geometric operator <-> means "distance between"

        END;
        $$ LANGUAGE plpgsql;"""
    cur.execute(sql_CreateFunction)
    sql_queryNode = "SELECT * FROM get_nearest_node(%s,%s)" %(lon,lat)
    nearestNode = pd.read_sql_query(sql_queryNode,con=conn)
    return nearestNode


def drivingDistance(startNode,distance,mapPath="./results/mapwithdrivedistance.html"):
    """ Returns all nodes and edges within a specified drive distance

    Detailed description

    Args:
        startNode (int): The id of your point
        distance (float): How far from this point you want to be able to reach
        mapPath (str): Where to write the map to

    Returns:
        nodes_features (folium features):
        edges_features (folium features):

    Raises:
        None

    Todo:
        * remove mapPath and convert to its own function
        * return geodataframe(s)
        * replace startNode with coordinates
        * figure out units for distance
        * implement drive time

    Tests:
        None
    """
    sql = "SELECT * FROM pgr_drivingDistance('SELECT id, source, target, cost, reverse_cost FROM ways',%s,%s);" %(startNode,distance)
    drivingDistance = pd.read_sql_query(sql,con=conn)
    nodes = list(drivingDistance['node'])
    nodes = [str(node) for node in nodes]
    edges = list(drivingDistance['edge'])
    edges = [str(edge) for edge in edges]
    sql_nodes = "select * from ways_vertices_pgr where id in (%s)" %(",".join(nodes))
    df_nodes = gpd.read_postgis(sql_nodes,con=conn,geom_col="the_geom")
    sql_edges = "select * from ways where id in (%s)" %(",".join(edges))
    df_edges = gpd.read_postgis(sql_edges,con=conn,geom_col="the_geom")
    df_nodes.crs = {'init':'epsg:4326'}
    df_edges.crs = {'init':'epsg:4326'}
    nodes_gjson = df_nodes.to_crs(epsg='4326').to_json()
    edges_gjson = df_edges.to_crs(epsg='4326').to_json()
    nodes_features = folium.features.GeoJson(nodes_gjson)
    edges_features = folium.features.GeoJson(edges_gjson)
    map2 = folium.Map( tiles='stamentoner', zoom_start=6)
    #map2.add_child(nodes_features)
    map2.add_child(edges_features)
    map2.save(mapPath)
    return nodes_features, edges_features


def kMultipleRoutes(startNode,endNode,k):
    """ Utilizes pgrouting's k-multiple routes

    Does not necessarily return routes which are very different from each other

    Args:
        startNode (int): Starting location
        endNode (int): Ending location
        k (int): number of routes to return

    Returns:
        None

    Raises:
        None

    Todo:
        * remove mapPath and convert to its own function
        * return geodataframe(s)
        * replace nodes with coordinates

    Tests:
        None
    """
    sql = "SELECT * FROM pgr_ksp('SELECT id, source, target, cost, reverse_cost FROM ways',%s,%s,%s);" %(startNode,endNode,k)
    df = pd.read_sql_query(sql,con=conn)
    edges = list(df['edge'])
    edges = [str(edge) for edge in edges]
    sql2 = "select * from ways where id in (%s)" %(",".join(edges))
    df2 = gpd.read_postgis(sql2,con=conn,geom_col="the_geom")
    map = folium.Map( tiles='stamentoner', zoom_start=6)
    df2.crs = {'init':'epsg:4326'}
    gjson = df2.to_crs(epsg='4326').to_json()
    lines = folium.features.GeoJson(gjson)
    map.add_child(lines)
    map.save('./results/mapwithmultipleroutes.html')


def kMultipleRoutes_LatLon(startLon,startLat,endLon,endLat,k):
    """ Utilizes pgrouting's k-multiple routes using start and end coordinates

    Does not necessarily return routes which are very different from each other

    Args:
        startLon (float): Longitude of starting point in EPSG:4326
        startLat (float): Latitude of starting point in EPSG:4326
        endLon (float): Longitude of ending point in EPSG:4326
        endLat (float): Latitude of ending point in EPSG:4326
        k (int): number of routes to return

    Returns:
        None

    Raises:
        None

    Todo:
        * Allow for projections
        * remove mapPath and convert to its own function
        * return geodataframe(s)

    Tests:
        None
    """
    startNode = int(get_nearest_node(startLon,startLat)["node_id"])
    endNode = int(get_nearest_node(endLon,endLat)["node_id"])
    sql = "SELECT * FROM pgr_ksp('SELECT id, source, target, cost, reverse_cost FROM ways',%s,%s,%s);" %(startNode,endNode,k)
    df = pd.read_sql_query(sql,con=conn)
    edges = list(df['edge'])
    edges = [str(edge) for edge in edges]
    sql2 = "select * from ways where id in (%s)" %(",".join(edges))
    df2 = gpd.read_postgis(sql2,con=conn,geom_col="the_geom")
    map = folium.Map( tiles='stamentoner', zoom_start=6)
    df2.crs = {'init':'epsg:4326'}
    gjson = df2.to_crs(epsg='4326').to_json()
    lines = folium.features.GeoJson(gjson)
    map.add_child(lines)
    map.save('./results/mapwithmultipleroutes.html')


def route(startLon,startLat,endLon,endLat):
    """ Routes between two points

    Takes longitude and latitude for two points and provides the pgrouting
    shortest route between the two, using Dijkstra's algorithm

    Args:
        startLon (float): Longitude of starting point in EPSG:4326
        startLat (float): Latitude of starting point in EPSG:4326
        endLon (float): Longitude of ending point in EPSG:4326
        endLat (float): Latitude of ending point in EPSG:4326

    Returns:
        None

    Raises:
        None

    Todo:
        * Allow for projections
        * remove mapPath and convert to its own function
        * return geodataframe(s)

    Tests:
        None
    """
    startNode = int(get_nearest_node(startLon,startLat)["node_id"])
    endNode = int(get_nearest_node(endLon,endLat)["node_id"])
    sql = "select * from pgr_dijkstra('select id, source, target, cost, reverse_cost FROM ways',%s,%s);" %(startNode,endNode)
    cur.execute(sql)
    rows = cur.fetchall()
    df = pd.read_sql_query(sql,con=conn)
    edges = list(df['edge'])
    edges = [str(edge) for edge in edges]
    sql2 = "select * from ways where id in (%s)" %(",".join(edges))
    df2 = gpd.read_postgis(sql2,con=conn,geom_col="the_geom")
    map = folium.Map( tiles='stamentoner', zoom_start=6)
    df2.crs = {'init':'epsg:4326'}
    gjson = df2.to_crs(epsg='4326').to_json()
    lines = folium.features.GeoJson(gjson)
    map.add_child(lines)
    map.save('./results/mapwithroute.html')


def routeWithAvoidance(startLon,startLat,endLon,endLat,linkIDsToAvoid=[]):
    """ Summary line

    Detailed description

    Args:
        param1 (int): The first parameter.
        param1 (str): The second parameter.

    Returns:
        network (pandas dataframe): The return and how to interpret it

    Raises:
        IOError: An error occured accessing the database

    Tests:
        >>> get_nearest_node(-92.1647,37.7252)
        node_id = 634267, dist = 124
    """
    startNode = int(get_nearest_node(startLon,startLat)["node_id"])
    endNode = int(get_nearest_node(endLon,endLat)["node_id"])
    linkIDsToAvoid = [str(x) for x in linkIDsToAvoid]
    linksToAvoidString = ','.join(linkIDsToAvoid)
    sql = "select * from pgr_dijkstra('select id, source, target, cost, reverse_cost FROM ways WHERE id NOT IN (%s)',%s,%s);" %(linksToAvoidString,startNode,endNode)
    cur.execute(sql)
    rows = cur.fetchall()
    df = pd.read_sql_query(sql,con=conn)
    edges = list(df['edge'])
    edges = [str(edge) for edge in edges]
    sql2 = "select * from ways where id in (%s)" %(",".join(edges))
    df2 = gpd.read_postgis(sql2,con=conn,geom_col="the_geom")
    map = folium.Map( tiles='stamentoner', zoom_start=6)
    df2.crs = {'init':'epsg:4326'}
    gjson = df2.to_crs(epsg='4326').to_json()
    lines = folium.features.GeoJson(gjson)
    map.add_child(lines)
    map.save('./results/mapwithrouteavoidance.html')
    return df2


def queryRoadAndPutMarkerOnMidPoint(roadID):
    """ Summary line

    Detailed description

    Args:
        param1 (int): The first parameter.
        param1 (str): The second parameter.

    Returns:
        network (pandas dataframe): The return and how to interpret it

    Raises:
        IOError: An error occured accessing the database

    Tests:
        >>> get_nearest_node(-92.1647,37.7252)
        node_id = 634267, dist = 124
    """
    return 0


## CURRENT TEST

## TESTS
def testGetNearestNode():
    nearestNode = get_nearest_node(-92.1647,37.7252) # Fort Leonard Wood, the negative is really important!
    return nearestNode

def testShorterQueryWithoutDistance:
    nearestNode = shorterQueryWithoutDistance(-92.1647,37.7252) # Fort Leonard Wood, the negative is really important!
    nearestNode


def testDrivingDistance():
    nodes_features, edges_features = drivingDistance(634267,0.1)
    return nodes_features, edges_features

def testKMultipleRoutes():
    kMultipleRoutes(634267,3,2)

def testKMultipleRoutes_LatLon():
    kMultipleRoutes_LatLon(-92.068859,37.846720,-92.142373,37.557935,100)

def test_route():
    route(-92.068859,37.846720,-92.142373,37.557935)

def testMapWithRoute():
    connString = "dbname='routing' user='postgres' host='localhost' password='postgres'"
    conn = psycopg2.connect(connString)
    cur = conn.cursor()
    sql = "select * from pgr_dijkstra('select id, source, target, cost, reverse_cost FROM ways',634267,3);"
    cur.execute(sql)
    rows = cur.fetchall()
    df = pd.read_sql_query(sql,con=conn)
    df
    edges = list(df['edge'])
    edges = [str(edge) for edge in edges]
    sql2 = "select * from ways where id in (%s)" %(",".join(edges))
    df2 = gpd.read_postgis(sql2,con=conn,geom_col="the_geom")
    df2
    df2.plot()
    map = folium.Map( tiles='stamentoner', zoom_start=6)
    df2.crs = {'init':'epsg:4326'browser}
    gjson = df2.to_crs(epsg='4326').to_json()
    lines = folium.features.GeoJson(gjson)
    map.add_child(lines)
    map.save('./results/mapwithroute.html')

def testRouteWithTurnsAndPopups():
    popups,locations = [], []
    for idx,row in df2.iterrows():
        locations.append([row['y1'],row['x1']])
        name = row['name']
        popups.append(name)

    h = folium.FeatureGroup(name="route")
    h.add_child(MarkerCluster(locations=locations,popups=popups))
    map.add_child(h)
    map.save('./results/mapwithrouteandpopups.html')

def testRoutingWithAvoidances():
    avoidanceIDs = [690751,690752,1302738,702378,709829]
    df2 = routeWithAvoidance(-92.068859,37.846720,-92.142373,37.557935,avoidanceIDs)
    df2.sort_values(by='length',ascending=False)
    return df2
