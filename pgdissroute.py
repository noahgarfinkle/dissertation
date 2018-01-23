import psycopg2
import geopandas as gpd
import pandas as pd
import folium
import folium.plugins as plugins

%matplotlib inline

def get_nearest_node(lon,lat):
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

nearestNode = get_nearest_node(-92.1647,37.7252) # Fort Leonard Wood, the negative is really important!
nearestNode

def shorterQueryWithoutDistance(lon,lat):
    sql = "SELECT * FROM ways_vertices_pgr ORDER BY the_geom <-> ST_GeometryFromText('POINT(%s %s)',4326) LIMIT 1;" %(lon,lat)
    nearestNode = pd.read_sql_query(sql,con=conn)
    return nearestNode

nearestNode = shorterQueryWithoutDistance(-92.1647,37.7252) # Fort Leonard Wood, the negative is really important!
nearestNode


def drivingDistance(startNode,distance):
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
    # https://ocefpaf.github.io/python4oceanographers/blog/2015/12/14/geopandas_folium/
    #map2.add_child(nodes_features)
    map2.add_child(edges_features)
    map2.save('./results/mapwithdrivedistance.html')
    return nodes_features, edges_features

nodes_features, edges_features = drivingDistance(634267,0.1)

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
map = folium.Map([-92.1647,37.7252], tiles='stamentoner', zoom_start=6)
# https://ocefpaf.github.io/python4oceanographers/blog/2015/12/14/geopandas_folium/
df2.crs = {'init':'epsg:4326'}
gjson = df2.to_crs(epsg='4326').to_json()
lines = folium.features.GeoJson(gjson)
map.add_child(lines)
map.save('./results/mapwithroute.html')

from folium.plugins import MarkerCluster
popups,locations = [], []
for idx,row in df2.iterrows():
    locations.append([row['y1'],row['x1']])
    name = row['name']
    popups.append(name)

h = folium.FeatureGroup(name="route")
h.add_child(MarkerCluster(locations=locations,popups=popups))
map.add_child(h)
map.save('./results/mapwithrouteandpopups.html')

def createCustomCost():
    return 0
