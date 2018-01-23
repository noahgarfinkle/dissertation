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
map = folium.Map([48., 5.], tiles='stamentoner', zoom_start=6)
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
