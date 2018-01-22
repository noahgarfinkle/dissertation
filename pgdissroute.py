import psycopg2
import geopandas as gpd
import pandas as pd

connString = "dbname='routing' user='postgres' host='localhost' password='postgres'"
conn = psycopg2.connect(connString)
cur = conn.cursor()
sql = "select * from pgr_dijkstra('select id, source, target, cost, reverse_cost FROM ways',2,3);"
cur.execute(sql)
rows = cur.fetchall()
df = pd.read_sql_query(sql,con=conn)
#df2 = gpd.read_postgis(sql,con=conn,geom_col="the_geom")
df
edges = list(df['edge'])
