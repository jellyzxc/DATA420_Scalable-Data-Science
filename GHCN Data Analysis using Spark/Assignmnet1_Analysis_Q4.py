import sys
from pyspark import SparkContext
from pyspark.sql import SparkSession
from pyspark.sql.types import * 
from pyspark.sql import functions as F

spark = SparkSession.builder.getOrCreate()
sc = SparkContext.getOrCreate()
#================== Q4 ====================--
 
#start_pyspark_shell -e 8 -c 4 -w 4 -m 4 
#start_pyspark_shell -e 32 -c 2 -w 4 -m 4 
 
#schema_Daily
schema_Daily = StructType([
    StructField('ID', StringType()),
    StructField('DATE', StringType()),
    StructField('ELEMENT', StringType()),
    StructField('VALUE', IntegerType()),
    StructField('MEASUREMENT_FLAG', StringType()),
    StructField('QUALITY_FLAG', StringType()),
    StructField('SOURCE_FLAG', StringType()),
    StructField('OBSERVATION_TIME', StringType()),
])

#---------------(a)------------2928664523 ------
path = "hdfs:///data/ghcnd/daily/*.csv.gz"
daily= (
    spark.read.format("com.databricks.spark.csv")
    .option("header", "false")
    .option("inferSchema", "false")
    .schema(schema_Daily)
    .load(path)
)
daily.count()   #2928664523
daily.rdd.getNumPartitions()  #108     load is a job here  and get the 108  Partitions

 
 

path = "hdfs:///data/ghcnd/daily/*.csv.gz"

rdd_daily = sc.textFile(path)
rdd_daily.getNumPartitions() #258

daily_4 = spark.read.csv(rdd_daily) 
daily_4.rdd.getNumPartitions() #258
 
 
 
# repartition is not necessary here  
daily2=daily.repartition(80)
daily2.count()   #2928664523   

daily3=daily.repartition(256)
daily3.count()   #2928664523


#-----try read each files and union them together to check the NumPartitions
#----I guess  there will  be  258 partitions


# command line : pip install hdfs
#Permission denied: '/usr/local/anaconda3/lib/python3.7/site-packages/docopt.py'

#try another way like below  and it works
# pip install hdfs --target=/users/home/xzh216/pythonPackage

import sys;
sys.path.append("/users/home/xzh216/pythonPackage/") 
import hdfs
from hdfs.client import Client
client = Client("http://node0:9870/")    
# echo $HADOOP_HOME    
# cat /opt/hadoop/hadoop/etc/hadoop/core-site.xml     --->node0
#node0  hostname    2.X50070   3.X 9870
#print("hdfs:", client.list(hdfs_path="/",status=True))
fileList=client.list("/data/ghcnd/daily/")

#create empty rdd
result = spark.createDataFrame(sc.emptyRDD(),schema_Daily)
path_pre = "hdfs:///data/ghcnd/daily/"
for file in fileList:
    daily_temp= (
        spark.read.format("com.databricks.spark.csv")
        .option("header", "false")
        .option("inferSchema", "false")
        .schema(schema_Daily)
        .load(path_pre+file)
    )  
    result=result.union(daily_temp)

result.rdd.getNumPartitions()    #258
result.count()#2928664523

 


#---------------(b)------------------
core_elements_list = ['PRCP', 'SNOW', 'SNWD', 'TMAX', 'TMIN']
daily_with_coreelements = daily.filter(F.col('ELEMENT').isin(core_elements_list))    

daily_with_coreelements.count()  #  2509690508

daily_with_coreelements.rdd.getNumPartitions()  #108 

#repartition(96)
daily=daily.repartition(96)
daily_with_coreelements = daily.filter(F.col('ELEMENT').isin(core_elements_list))    

daily_with_coreelements.count()  #  2509690508

daily_with_coreelements.rdd.getNumPartitions()  #80  

#repartition(128)
daily=daily.repartition(128)
daily_with_coreelements = daily.filter(F.col('ELEMENT').isin(core_elements_list))    

daily_with_coreelements.count()   
daily_with_coreelements.rdd.getNumPartitions()   


#repartition(256)
daily=daily.repartition(256)
daily_with_coreelements = daily.filter(F.col('ELEMENT').isin(core_elements_list))    

daily_with_coreelements.count()   

daily_with_coreelements.rdd.getNumPartitions()   


#repartition(108)-back 

daily=daily.repartition(108)
daily_with_coreelements = daily.filter(F.col('ELEMENT').isin(core_elements_list))    
daily_with_coreelements.count()

from pyspark import StorageLevel
daily_with_coreelements.persist(StorageLevel.MEMORY_AND_DISK).count()
#daily_with_coreelements.unpersist() 
 
 

# How many observations are there for each of the five core elements
# no distinct  ---
(
    daily_with_coreelements
    .select(F.col('ID'),F.col('ELEMENT'))
    .groupby('ELEMENT')
    .agg(F.count('ID').alias('Observations'))
    .orderBy(F.col("Observations").desc())
).show(5,False)

# the same as before
(
    daily_with_coreelements
    .groupby('ELEMENT').count()
    .orderBy(F.col("count").desc())
 ).show(5,False)

 
schema_Inventory = StructType([
    StructField('ID', StringType()),
    StructField('LATITUDE', DoubleType()),   
    StructField('LONGITUDE', DoubleType()),
    StructField('ELEMENT', StringType()),
    StructField('FIRSTYEAR', IntegerType()),
    StructField('LASTYEAR', IntegerType()), 
])

inventory_text = (
    spark.read.format("text")
    .load("hdfs:///data/ghcnd/inventory")
)
inventory = inventory_text.select(
    F.trim(F.substring(F.col('value'), 1, 11)).alias('ID').cast(schema_Inventory['ID'].dataType),
    F.trim(F.substring(F.col('value'), 13, 8)).alias('LATITUDE').cast(schema_Inventory['LATITUDE'].dataType),
    F.trim(F.substring(F.col('value'), 22, 9)).alias('LONGITUDE').cast(schema_Inventory['LONGITUDE'].dataType),
    F.trim(F.substring(F.col('value'), 32, 4)).alias('ELEMENT').cast(schema_Inventory['ELEMENT'].dataType),
    F.trim(F.substring(F.col('value'), 37, 4)).alias('FIRSTYEAR').cast(schema_Inventory['FIRSTYEAR'].dataType),
    F.trim(F.substring(F.col('value'), 42, 4)).alias('LASTYEAR').cast(schema_Inventory['LASTYEAR'].dataType)
)
inventory_with_coreelements = inventory.filter(F.col('ELEMENT').isin(core_elements_list)) 
inventory_with_coreelements.cache()
# How many inventores are there for each of the five core elements
(
    inventory_with_coreelements
    .select(F.col('ID'),F.col('ELEMENT'))
    .groupby('ELEMENT')
    .agg(F.count('ID').alias('count'))
    .orderBy(F.col("count").desc())
).show(5,False)



#How many different stations are there for each of the five core elements
(
    daily_with_coreelements
    .select(F.col('ID'),F.col('ELEMENT'))
    .distinct()
    .groupby('ELEMENT')
    .agg(F.count('ID').alias('Stations'))
    .orderBy(F.col("Stations").desc())
).show(5,False)


(
    inventory_with_coreelements
    .select(F.col('ID'),F.col('ELEMENT'))
    .distinct()
    .groupby('ELEMENT')
    .agg(F.count('ID').alias('Stations'))
    .orderBy(F.col("Stations").desc())
).show(5,False)



 

stations = spark.read.orc("hdfs:////user/xzh216/Assign1/output/stations_enriched.orc")
#daily   stations  #check stations in daily  not in stations_enriched

#back to processing Q3 e  =>all  daily    

data_broadcast=(
    daily.
    join(F.broadcast(stations), daily.ID == stations.STATION_ID)
)

 
(data_broadcast
 .where((F.col("STATION_ID")=="")| (F.col("STATION_ID").isNull()))
 .select(F.col("ID"))
 .dropDuplicates()  #find the stations not occured in all daily    still 0 
 ).count()

#>> 64 cores   for   1.3 min 

#---------------(c)------------------
#---check:is there any element without value ?
daily_with_coreelements.count()   #2509690508
daily_with_coreelements.where((F.col("VALUE")=="")| (F.col("VALUE").isNull())).count()   #0

# --make sure all  observations have VALUES, So we can foucs on ID','DATE' and 'ELEMENT'
 
#selsect subset for  'TMAX', 'TMIN'
Temp_list = ['TMAX', 'TMIN']


daily_with_tempelements = daily_with_coreelements.filter(F.col('ELEMENT').isin(Temp_list))    
daily_with_tempelements.count()   #872005599
daily_with_tempelements.show(5,False)


#daily_temp = (
#    daily_with_tempelements
#    .groupby('ID','DATE')
#    .agg(F.collect_set('ELEMENT').alias('Temp'))    #collect_set   nor duplicate <ID,FDATA>  
#  ) 


#daily_temp.show(5,False)
#daily_temp.count()  #445138151
 

daily_temp = (
    daily_with_coreelements   #'TMAX', 'TMIN'  are  core elements
    .filter(F.col('ELEMENT').isin(Temp_list))
    .groupby('ID','DATE')
    .agg(F.collect_list('ELEMENT').alias('Temp'))    #collect_list    
  ) 
daily_temp.count() #445138151

#the same as above  so there is no duplicate <ID,DATA>  
  
daily_temp = ( 
     daily      #daily
    .filter(F.col('ELEMENT').isin(Temp_list))
    .groupby('ID','DATE')
    .agg(F.collect_list('ELEMENT').alias('Temp'))    #collect_set   nor duplicate <ID,FDATA>  
  )   
daily_temp.count()  #445138151

#So using daily_with_coreelements  is enough
 
  
#create new columns for sign of temp 

daily_temp_elements=(
  daily_temp 
 .withColumn('Has_TMAX', F.array_contains( F.col("Temp"), "TMAX"))  
 .withColumn('Has_TMIN', F.array_contains( F.col("Temp"), "TMIN"))
 )
daily_temp_elements.show(100,False)

#<ID DATE Temp Has_TMAX Has_TMIN>

#how many observations of TMIN do not have a corresponding observation of TMAX.
(
daily_temp_elements
.filter(F.col("Has_TMIN") & (F.col("Has_TMAX")==False))
).count()  # 8428801

#How many different stations contributed to these observations?
(
daily_temp_elements
.filter(F.col("Has_TMIN") & (F.col("Has_TMAX")==False))
.select(F.col("ID"))
.distinct()
).count()   #27526     total stations  115081




#only Has_TMIN or Has_TMAX    -way1
(
daily_temp_elements
.filter(F.size("Temp")==1)
).count()  #  18270703


#only Has_TMIN or Has_TMAX    -way2 
(
 daily_temp_elements 
.filter(((F.col("Has_TMIN")) & (F.col("Has_TMAX")))==False)
).count()  #   18270703



#How many different stations contributed to these observations?
(
daily_temp_elements
.filter( (F.col("Has_TMIN") & (F.col("Has_TMAX")))==False)
.select(F.col("ID"))
.distinct()
).count()   # 30177







#---------------(d)------------------ 
#Filter daily to obtain all observations of TMIN and TMAX for all stations in New Zealand, and save the result to your output directory
stations = spark.read.orc("hdfs:////user/xzh216/Assign1/output/stations_enriched.orc")
stations.show(5,False)

stations_NZ_ID = (stations
               .filter(F.col("FIPS")=='NZ')
               .distinct()
               .select(F.col('STATION_ID'))
               )   
stations_NZ_ID.count()  #15 stations in NZ
stations_NZ_ID.cache()
  


# way1  try in   -- 

stations_NZ_list=[row.STATION_ID for row in stations_NZ_ID.collect()]

#tip  collect()   Returns all the records as a list of Row.   and  isin(  list)
daily_with_tempelements_NZ=(
    daily_with_tempelements.filter(F.col('ID').isin(stations_NZ_list) )
)
daily_with_tempelements_NZ.count()   #458892
# 1.5 min  ----36s when persist daily_with_coreelements
 
#way2 try broadcast join    and  stations_NZ_ID.cache()

daily_with_tempelements.rdd.getNumPartitions()  #108

daily_with_tempelements_NZ=(
    daily_with_tempelements 
    .join(F.broadcast(stations_NZ_ID), daily.ID == stations_NZ_ID.STATION_ID)   #broadcast to 128 cores  so 128 tasks 
    .drop("STATION_ID")
)
daily_with_tempelements_NZ.count()  #458892
#1.6min  slow than way1    ----22s  when persist daily_with_coreelements

daily_with_tempelements_NZ.show()


daily_with_tempelements_NZ =(
    daily_with_tempelements_NZ
    .withColumn('YEAR', F.trim(F.substring(F.col('DATE'), 1, 4)).cast(StringType()))
    .withColumn('MONTH',F.trim(F.substring(F.col('DATE'), 5, 2)).cast(StringType()))
 )

daily_with_tempelements_NZ.rdd.getNumPartitions()   #108

outputpath = "hdfs:///user/xzh216/Assign1/output/"
daily_with_tempelements_NZ.write.format('orc').mode("overwrite").save(outputpath+'daily_with_tempelements_NZ.orc')
daily_with_tempelements_NZ.write.format('csv').mode("overwrite").save(outputpath+'daily_with_tempelements_NZ.csv')
 
# hdfs fsck /user/xzh216/Assign1/output/daily_with_tempelements_NZ.orc -blocks     
#  1.3047M	  81 block  IN HDFS

daily_with_tempelements_NZ=daily_with_tempelements_NZ.repartition(1)
daily_with_tempelements_NZ.rdd.getNumPartitions() #1
  
daily_with_tempelements_NZ.write.format('orc').mode("overwrite").save(outputpath+'daily_with_tempelements_NZ.orc')
daily_with_tempelements_NZ.write.format('csv').mode("overwrite").save(outputpath+'daily_with_tempelements_NZ.csv')
#daily_with_tempelements_NZ.orc     1.155956268M	 1 block


# how many years are covered by the observations?
daily_with_tempelements_NZ = spark.read.orc("hdfs:////user/xzh216/Assign1/output/daily_with_tempelements_NZ.orc")
daily_with_tempelements_NZ.show(5,False)

(daily_with_tempelements_NZ
.agg(F.max("YEAR"),F.min("YEAR"))
).show()

#|max(YEAR)|min(YEAR)|
#+---------+---------+
#|     2020|     1940|

(daily_with_tempelements_NZ
 .filter(F.col('YEAR')==1940)
 .agg(F.min("MONTH"))
 ).show()  #03
daily_with_tempelements_NZ.filter(F.col('YEAR')==2020).agg(F.max("MONTH")).show()  #03

#---------------(e)------------------ 
#Group the precipitation observations by year and country. Compute the average rainfall in each year for each country, 
#and save this result to your output directory
 
#PRCP = Precipitation (tenths of mm)    element
 
stations.cache()
stations.show()   #FIPS|   COUNTRY_NAME
stations.rdd.getNumPartitions()   #4
daily.rdd.getNumPartitions()  #108

daily.where( (F.col("ELEMENT")=="PRCP") & ((F.col("VALUE")=="")| (F.col("VALUE").isNull())) ).count()

daily_PRCPs= (
     daily      #daily
    .filter(F.col("ELEMENT")=="PRCP")
    .withColumn('YEAR', F.trim(F.substring(F.col('DATE'), 1, 4)).cast(StringType()))
)

daily_PRCPs.persist().count()   #1021682210


#0  no value is null so no need to remove NULL
daily_Avg_Rainfall = ( 
     daily_PRCPs
    .join(F.broadcast(stations), daily.ID == stations.STATION_ID)
    .groupby('FIPS','COUNTRY_NAME','YEAR')
    .agg(F.mean("VALUE").alias('Avg_Rainfall_daily'),F.count("VALUE").alias('Counts_observations'),F.sum("VALUE").alias('Sum_Rainfall'))
    .withColumn("Avg_Rainfall_yearly",F.col("Avg_Rainfall_daily")*365)
    .orderBy(F.col("Avg_Rainfall_daily").desc())
  ) 
#daily_Avg_Rainfall.count()   #16797 <219(COUNTRY)*258 YEAR 
daily_Avg_Rainfall.show(10,False)

#Why ??
PRCP_EK_2000=(
 daily_PRCPs
.join(F.broadcast(stations), daily.ID == stations.STATION_ID)
.filter((F.col('FIPS')=="EK") & (F.col('YEAR')=="2000"))
)
PRCP_EK_2000.count()   #1

#EKM00064810|20000622| PRCP| 4361|  null|  G|

( 
     daily_PRCPs
     .filter((F.col('QUALITY_FLAG')=="")|(F.col('QUALITY_FLAG').isNull()))
    .join(F.broadcast(stations), daily.ID == stations.STATION_ID)
    .groupby('FIPS','COUNTRY_NAME','YEAR')
    .agg(F.mean("VALUE").alias('Avg_Rainfall_daily'),F.count("VALUE").alias('Counts_observations'),F.sum("VALUE").alias('Sum_Rainfall'))
    .withColumn("Avg_Rainfall_yearly",F.col("Avg_Rainfall_daily")*365)
    .orderBy(F.col("Avg_Rainfall_daily").desc())
).show(10,False)

#BH  |Belize              |1978|1958.833333333333
PRCP_Belize_1978=(
 daily_PRCPs
.filter((F.col('QUALITY_FLAG')=="")|(F.col('QUALITY_FLAG').isNull()))
.join(F.broadcast(stations), daily.ID == stations.STATION_ID)
.filter((F.col('FIPS')=="BH") & (F.col('YEAR')=="1978"))
.select("ID", "DATE","ELEMENT","VALUE","MEASUREMENT_FLAG","QUALITY_FLAG")
)
PRCP_Belize_1978.count()

 
daily_Avg_Rainfall.select(F.col('FIPS'),F.col('COUNTRY_NAME')).distinct().count()  #218

countries = spark.read.orc("hdfs:////user/xzh216/Assign1/output/countries.orc")
countries.select(F.col('CODE'),F.col('NAME')).distinct().count()   #219

(countries
 .join(daily_Avg_Rainfall, countries.CODE==daily_Avg_Rainfall.FIPS, how='left')
 .drop("StationCount","Year","Avg_Rainfall")
 .distinct()
 .where(F.col('FIPS').isNull())
 ).show(1,False)    #PC  |Pitcairn Islands [United Kingdom]
     
stations.show(5)
countries.show(5)

(countries
.join(stations, countries.CODE==stations.FIPS, how='left')
.select(F.col('CODE'),F.col('NAME'),F.col('COUNTRY_NAME'))
 .distinct()
 .where(F.col('COUNTRY_NAME').isNull())
).count() #0

 
( 
 daily_PRCPs
.join(F.broadcast(stations), daily_PRCPs.ID == stations.STATION_ID)
.where(F.col('FIPS')=='PC')
).count()  # 0   the country PC NO PRCP
 

( 
 daily
.join(F.broadcast(stations), daily.ID==stations.STATION_ID)
.where(F.col('FIPS')=='PC')
).count()  # 19930
 

( 
 daily
.join(F.broadcast(stations), daily.ID==stations.STATION_ID)
.where(F.col('FIPS')=='PC')
.groupby('ELEMENT')
.agg(F.count("ID").alias('ELEMENTCounts'))   
).show(100,False)



#not broadcast
daily_Avg_Rainfall_2 = ( 
     daily_PRCPs
    .join(stations, daily.ID == stations.STATION_ID, how="left")
    .groupby('FIPS','COUNTRY_NAME','YEAR')
    .agg(F.mean("VALUE").alias('Avg_Rainfall'))   
    .orderBy(F.col("Avg_Rainfall").desc())
  ) 
#daily_Avg_Rainfall_2.count()
daily_Avg_Rainfall_2.show(5,False)

 

daily_Avg_Rainfall_2.rdd.getNumPartitions()


daily_Avg_Rainfall.count()  #16797


daily_Avg_Rainfall.rdd.getNumPartitions()
daily_Avg_Rainfall=daily_Avg_Rainfall.repartition(1)
daily_Avg_Rainfall.rdd.getNumPartitions()



outputpath = "hdfs:///user/xzh216/Assign1/output/"
daily_Avg_Rainfall.write.format('orc').mode("overwrite").save(outputpath+'daily_Avg_Rainfall.orc')
daily_Avg_Rainfall.write.format('csv').mode("overwrite").save(outputpath+'daily_Avg_Rainfall.csv')

# cumulative rainfall for each country

from pyspark.sql import Window   
windowSpec = Window.partitionBy('FIPS','COUNTRY_NAME').orderBy("YEAR")
daily_Cum_Rainfall= (
  daily_Avg_Rainfall 
  .withColumn('cumsum_Rainfall_byAvgDaily', F.sum(daily_Avg_Rainfall['Avg_Rainfall_daily']).over(windowSpec))
  .withColumn('cumsum_Rainfall_byAvgYearly', F.sum(daily_Avg_Rainfall['Avg_Rainfall_yearly']).over(windowSpec))
  .withColumn('cumsum_Rainfall_bySum', F.sum(daily_Avg_Rainfall['Sum_Rainfall']).over(windowSpec))
  )
                                                                                            
daily_Cum_Rainfall.show(1000)

daily_Cum_Rainfall=daily_Cum_Rainfall.repartition(1)
daily_Cum_Rainfall.write.format('csv').mode("overwrite").save(outputpath+'daily_Cum_Rainfall.csv')
daily_Cum_Rainfall.write.format('orc').mode("overwrite").save(outputpath+'daily_Cum_Rainfall.orc')

 