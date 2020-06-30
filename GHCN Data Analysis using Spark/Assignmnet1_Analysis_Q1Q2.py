import sys
from pyspark import SparkContext
from pyspark.sql import SparkSession
from pyspark.sql.types import * 
from pyspark.sql import functions as F

spark = SparkSession.builder.getOrCreate()
sc = SparkContext.getOrCreate()
#================== Q1 ====================--

#read data from my output folder (a) How many stations are there in total? How many stations have been active in 2020? 
stations = spark.read.orc("hdfs:////user/xzh216/Assign1/output/stations_enriched.orc")
stations.show(5,False)
stations.count()

#(a) 
#How many stations are there in total? 
stations.select(F.col("STATION_ID")).distinct().count()  
#---115081

#How many stations have been active in 2020?
stations.select("STATION_ID","OPEN","CLOSE").show(10,False)
 
stations.where((F.col("CLOSE")>=2020) &(F.col("OPEN")<=2020) ).distinct().count()
#---32850

#How many stations are in each of the GCOS Surface Network (GSN), the US Historical Climatology Network (HCN), and the US Climate Reference Network (CRN)? 
stations.select(F.col("GSN_FLAG")).distinct().show()  
stations.where(F.col("GSN_FLAG")=="GSN").count()   #991
stations.where(F.col("HCN/CRN_FLAG")=="HCN").count()  #1218
stations.where(F.col("HCN/CRN_FLAG")=="CRN").count()  #233

stations_GSN=(
  stations
  .where(F.col("GSN_FLAG")=="GSN")
  .groupby("GSN_FLAG")
  .agg(F.count('STATION_ID').alias('sum_stations'))
  .orderBy(F.col("sum_stations").desc())
)
stations_GSN.show(5,False)
 
 
stations_HCN_CRN=(
  stations
  .where((F.col("HCN/CRN_FLAG").isNotNull())&(F.col("HCN/CRN_FLAG")!=""))
  .groupby("HCN/CRN_FLAG")
  .agg(F.count('STATION_ID').alias('sum_stations'))
  .orderBy(F.col("sum_stations").desc())
)
stations_HCN_CRN.show(5,False) 
 
#Are there any stations that are in more than one of these networks?
(stations 
 .withColumn("GSN", F.when(stations["GSN_FLAG"] == "GSN", 1).otherwise(0))
 .withColumn("HCN", F.when(stations["HCN/CRN_FLAG"] == "HCN", 1).otherwise(0))
 .withColumn("CRN", F.when(stations["HCN/CRN_FLAG"] == "CRN", 1).otherwise(0))
 .withColumn("SUM", F.col("GSN")+ F.col("HCN")+F.col("CRN"))
 .filter(F.col("SUM")>1)
 ).count()

(stations 
 .withColumn("GSN", F.when(stations["GSN_FLAG"] == "GSN", 1).otherwise(0))
 .withColumn("HCN", F.when(stations["HCN/CRN_FLAG"] == "HCN", 1).otherwise(0))
 .withColumn("CRN", F.when(stations["HCN/CRN_FLAG"] == "CRN", 1).otherwise(0))
 .withColumn("SUM", F.col("GSN")+ F.col("HCN")+F.col("CRN"))
 .filter(F.col("SUM")>1)
 .select(F.col("STATION_ID"),F.col("STATION_NAME"),F.col("GSN_FLAG"),F.col("HCN/CRN_FLAG"),F.col("SUM"))
 ).show(14,False)

#----14
 
#(b) 
#Count the total number of stations in each country, and store the output in countries using the withColumnRenamed command.

# get the count
countries_ex=(
    stations
    .groupby('FIPS')
    .agg(F.count('STATION_ID'))
)
countries_ex.count()
# join with countries
countries_text = (
    spark.read.format("text")
    .load("hdfs:///data/ghcnd/countries")
)
schema_Countries = StructType([
    StructField('CODE', StringType()),
    StructField('NAME', StringType()),
])
countries = countries_text.select(
    F.trim(F.substring(F.col('value'), 1, 2)).alias('CODE').cast(schema_Countries['CODE'].dataType),   #1-2
    F.trim(F.substring(F.col('value'), 4, 47)).alias('NAME').cast(schema_Countries['NAME'].dataType)   #4-50  3 space
)
#atttch the column 
countries = (
    countries
        .join(countries_ex, countries.CODE==countries_ex.FIPS, how='left')
        .drop('FIPS')
        .withColumnRenamed('count(STATION_ID)', 'StationCount')
)
countries.show()
countries.count()  #make sure  219 countries




#save countries
 
outputpath = "hdfs:///user/xzh216/Assign1/output/"
countries.write.format('orc').mode("overwrite").save(outputpath+'countries.orc')
 
#Do the same for states and save a copy of each table to your output directory.

#`avg`, `max`, `min`, `sum`, `count`  -agg
states_ex=(
    stations
    .groupby('STATE_CODE')
    .agg(F.count('STATION_ID'))   #stations.groupBy("STATE_CODE").count()
)
states_ex.count()
# 54


schema_States = StructType([
    StructField('CODE', StringType()),
    StructField('NAME', StringType())
])
states_text = (
    spark.read.format("text")
    .load("hdfs:///data/ghcnd/states")
)
states = states_text.select(
    F.trim(F.substring(F.col('value'), 1, 2)).alias('CODE').cast(schema_States['CODE'].dataType),
    F.trim(F.substring(F.col('value'), 4, 47)).alias('NAME').cast(schema_States['NAME'].dataType)
) 

states = (
    states
        .join(states_ex, states.CODE==states_ex.STATE_CODE, how='left')
        .drop('STATE_CODE')
        .withColumnRenamed('count(STATION_ID)', 'StationCount')
)
states.show()
states.count()    #74
 
states.filter(F.col("StationCount").isNull()).count()
states.filter(F.col("StationCount").isNull()).show(21,False) 
# 21 states without stations 
states.write.format('orc').mode("overwrite").save(outputpath+'states.orc')

#(c) 
#How many stations are there in the Northern Hemisphere only?
stations.filter(F.col("LATITUDE")>0).distinct().count() #89745
stations.filter(F.col("LATITUDE")<0).distinct().count()  #25336


#Some of the countries in the database are territories of the United States as indicated by the name of the country. 
#How many stations are there in total in the territories of the United States around the world? 
countries.show(10,False)
territories_us = (
          countries.filter(countries.NAME.like('%United States%'))
          .filter(F.col("CODE")!="US")
          .orderBy(F.col("StationCount"))
)
territories_us.show()                 

#way1  using countries get from Q1(b)
territories_us.select(F.col("StationCount")).agg(F.sum('StationCount')).show()  #316

 
#way2  filter first  then join
(stations
 .join(territories_us, stations.FIPS==territories_us.CODE, how='inner')   # inener 
 ).count()
#62183-61867





#================== Q2 ====================--
#LATITUDE   hor
#LONGITUDE  ver
# define native python function
from math import radians, degrees, sin, cos, asin, acos, sqrt
def StationsDistance_1(s_lon,s_lat,t_lon,t_lat):
          s_lon,s_lat,t_lon,t_lat= map(radians, [s_lon,s_lat,t_lon,t_lat])
          return 6371 * (acos(sin(s_lat) * sin(t_lat) + cos(s_lat) * cos(t_lat) * cos(s_lon - t_lon)))
 
 
from math import radians, sin, cos, asin, sqrt
def StationsDistance_2(lon1, lat1, lon2, lat2):
          lon1, lat1, lon2, lat2 = map(radians, [lon1, lat1, lon2, lat2])
          dlon = lon2 - lon1
          dlat = lat2 - lat1
          a = sin(dlat / 2) ** 2 + cos(lat1) * cos(lat2) * sin(dlon / 2) ** 2
          return 2 * 6371 * asin(sqrt(a)) 
 
 
#----- Permission denied:
#import subprocess
#def install(package):
#          subprocess.check_call([sys.executable, "-m", "pip", "install", package])
          
#install("haversine")         
#from haversine import haversine, Unit
#def StationsDistance_3(s_lon,s_lat,t_lon,t_lat): 
#          point1 = (s_lat, s_lon)
#          point2 = (t_lat, t_lon)
#          return haversine(point1, point2) 

# wrappe it to pyspark sql api     
#pyspark.sql.functions.udf(f=None, returnType=StringType)
StationsDistance_1_udf = F.udf(StationsDistance_1, DoubleType())
StationsDistance_2_udf = F.udf(StationsDistance_2, DoubleType())
#StationsDistance_3_udf = F.udf(StationsDistance_3, DoubleType())
 
 
#test this function by using CROSS JOIN on a small subset of stations to generate a table with two stations in each row.
 
test = stations.filter((F.col('OPEN')==2019) & (F.col('CLOSE')>=2020)).select(F.col('STATION_ID'), F.col('LATITUDE'), F.col('LONGITUDE'))
test.count()  #717
test_1 = (
         test
        .withColumnRenamed('STATION_ID', 'ID_1')
        .withColumnRenamed('LATITUDE', 'LATITUDE_1')
        .withColumnRenamed('LONGITUDE', 'LONGITUDE_1')
)

test_2 = (
        test
        .withColumnRenamed('STATION_ID', 'ID_2')
        .withColumnRenamed('LATITUDE', 'LATITUDE_2')
        .withColumnRenamed('LONGITUDE', 'LONGITUDE_2')
)
 
test_pairs = (
         test_1
        .crossJoin(test_2)
        .filter(F.col('ID_1') != F.col('ID_2'))   # or will cause 0 distance
)

(test_pairs
        .withColumn('distance',  StationsDistance_1_udf('LATITUDE_1', 'LONGITUDE_1', 'LATITUDE_2', 'LONGITUDE_2'))
).show()
 
(test_pairs
        .withColumn('distance',  StationsDistance_2_udf('LATITUDE_1', 'LONGITUDE_1', 'LATITUDE_2', 'LONGITUDE_2'))
).show()
  
# two functions get the same result  
 
 
# compute the pairwise distances between all stations in New Zealand,and save the result to your output directory

stations_NZ = (stations.
               filter(F.col("FIPS")=='NZ')
               .select(F.col('STATION_ID'), F.col('LATITUDE'), F.col('LONGITUDE'))
               )   #15 stations in NZ
stations_NZ.count()

stations_NZ_1 = (
         stations_NZ
        .withColumnRenamed('STATION_ID', 'ID_1')
        .withColumnRenamed('LATITUDE', 'LATITUDE_1')
        .withColumnRenamed('LONGITUDE', 'LONGITUDE_1')
)

stations_NZ_2 = (
          stations_NZ
        .withColumnRenamed('STATION_ID', 'ID_2')
        .withColumnRenamed('LATITUDE', 'LATITUDE_2')
        .withColumnRenamed('LONGITUDE', 'LONGITUDE_2')
)

stations_NZ_pairs = (
         stations_NZ_1
        .crossJoin(stations_NZ_2)
        .filter(F.col('ID_1') != F.col('ID_2'))   # or will cause 0 distance
)
StationsDistance_NZ=(stations_NZ_pairs
         .withColumn('distance',  StationsDistance_1_udf('LATITUDE_1', 'LONGITUDE_1', 'LATITUDE_2', 'LONGITUDE_2'))
)
StationsDistance_NZ.count()  #210 records   =15*14
#shoud be 105
StationsDistance_NZ=(
       StationsDistance_NZ.dropDuplicates(["distance"]).orderBy(F.col("distance").desc())
)
StationsDistance_NZ.count()  #105

#  ["distance"]   not  "distance"
outputpath = "hdfs:///user/xzh216/Assign1/output/"
StationsDistance_NZ.write.format('orc').mode("overwrite").save(outputpath+'StationsDistance_NZ.orc')

 #What two stations are the geographically furthest apart in New Zealand?
StationsDistance_NZ.head(1)
  
#Row(ID_1='NZ000093994', ID_2='NZ000939450', distance=2950.702144239996)

