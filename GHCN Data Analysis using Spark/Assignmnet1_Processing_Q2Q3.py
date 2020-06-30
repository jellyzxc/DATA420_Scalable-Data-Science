# Python and pyspark modules required
import sys
from pyspark import SparkContext
from pyspark.sql import SparkSession
from pyspark.sql.types import * 
from pyspark.sql import functions as F
# Required to allow the file to be submitted and run using spark-submit instead
# of using pyspark interactively
 
 
spark = SparkSession.builder.getOrCreate()
sc = SparkContext.getOrCreate()


#set parameters of resources
#2 executors, 1 core per executor, 1 GB of executor memory, and 1 GB  of master memory

#-------Q2-a----Define schemas -------------

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


# schema_Stations
schema_Stations = StructType([
    StructField('ID', StringType()),
    StructField('LATITUDE', DoubleType()),
    StructField('LONGITUDE', DoubleType()),
    StructField('ELEVATION', DoubleType()),
    StructField('STATE', StringType()),
    StructField('NAME', StringType()),
    StructField('GSN_FLAG', StringType()),
    StructField('HCN/CRN_FLAG', StringType()),
    StructField('WMO_ID', StringType()),
    
])
#schema_Countries
schema_Countries = StructType([
    StructField('CODE', StringType()),
    StructField('NAME', StringType()),
])
# schema_Inventory
schema_Inventory = StructType([
    StructField('ID', StringType()),
    StructField('LATITUDE', DoubleType()),   
    StructField('LONGITUDE', DoubleType()),
    StructField('ELEMENT', StringType()),
    StructField('FIRSTYEAR', IntegerType()),
    StructField('LASTYEAR', IntegerType()), 
])

#schema_States
schema_States = StructType([
  StructField('CODE', StringType()),
  StructField('NAME', StringType()),
])

#----Q2-b----load 1000 rows daily/2020.csv.gz---------------

daily = (
    spark.read.format("com.databricks.spark.csv")
    .option("header", "false")
    .option("inferSchema", "false")
    .schema(schema_Daily)
    .load("hdfs:///data/ghcnd/daily/2020.csv.gz")
    .limit(1000)
)
daily.cache() 
daily.show(5)


#----Q2-C----load metadata---------------
# load text
countries_text = (
    spark.read.format("text")
    .load("hdfs:///data/ghcnd/countries")
)
countries_text.show(5)

#parse
countries = countries_text.select(
    F.trim(F.substring(F.col('value'), 1, 2)).alias('CODE').cast(schema_Countries['CODE'].dataType),   #1-2
    F.trim(F.substring(F.col('value'), 4, 47)).alias('NAME').cast(schema_Countries['NAME'].dataType)   #4-50  3 space
)
countries.show(5)
countries.count()

 
inventory_text = (
    spark.read.format("text")
    .load("hdfs:///data/ghcnd/inventory")
)
inventory_text.show(5)
inventory = inventory_text.select(
    F.trim(F.substring(F.col('value'), 1, 11)).alias('ID').cast(schema_Inventory['ID'].dataType),
    F.trim(F.substring(F.col('value'), 13, 8)).alias('LATITUDE').cast(schema_Inventory['LATITUDE'].dataType),
    F.trim(F.substring(F.col('value'), 22, 9)).alias('LONGITUDE').cast(schema_Inventory['LONGITUDE'].dataType),
    F.trim(F.substring(F.col('value'), 32, 4)).alias('ELEMENT').cast(schema_Inventory['ELEMENT'].dataType),
    F.trim(F.substring(F.col('value'), 37, 4)).alias('FIRSTYEAR').cast(schema_Inventory['FIRSTYEAR'].dataType),
    F.trim(F.substring(F.col('value'), 42, 4)).alias('LASTYEAR').cast(schema_Inventory['LASTYEAR'].dataType)
)
inventory.show(5)
inventory.count()

 
states_text = (
    spark.read.format("text")
    .load("hdfs:///data/ghcnd/states")
)
states_text.show(5, False)
states = states_text.select(
    F.trim(F.substring(F.col('value'), 1, 2)).alias('CODE').cast(schema_States['CODE'].dataType),
    F.trim(F.substring(F.col('value'), 4, 47)).alias('NAME').cast(schema_States['NAME'].dataType)
)
states.show(5, False)
states.count()
 
 
 
stations_text = (
    spark.read.format("text")
    .load("hdfs:///data/ghcnd/stations")
)
stations_text.show(5)

stations = stations_text.select(
    F.trim(F.substring(F.col('value'), 1, 11)).alias('ID').cast(schema_Stations['ID'].dataType),
    F.trim(F.substring(F.col('value'), 13, 8)).alias('LATITUDE').cast(schema_Stations['LATITUDE'].dataType),
    F.trim(F.substring(F.col('value'), 22, 9)).alias('LONGITUDE').cast(schema_Stations['LONGITUDE'].dataType),
    F.trim(F.substring(F.col('value'), 32, 6)).alias('ELEVATION').cast(schema_Stations['ELEVATION'].dataType),
    F.trim(F.substring(F.col('value'), 39, 2)).alias('STATE_CODE').cast(schema_Stations['STATE'].dataType),
    F.trim(F.substring(F.col('value'), 42, 30)).alias('STATION_NAME').cast(schema_Stations['NAME'].dataType),
    F.trim(F.substring(F.col('value'), 73, 3)).alias('GSN_FLAG').cast(schema_Stations['GSN_FLAG'].dataType),
    F.trim(F.substring(F.col('value'), 77, 3)).alias('HCN/CRN_FLAG').cast(schema_Stations['HCN/CRN_FLAG'].dataType),
    F.trim(F.substring(F.col('value'), 81, 5)).alias('WMO_ID').cast(schema_Stations['WMO_ID'].dataType)
)
stations.show(5, False)
stations.count()

#How many stations do not have a WMO ID? 
(stations.select(F.col('ID'),F.col('WMO_ID'))
        .distinct()
        .filter(F.col('WMO_ID')=="")
 ).count()   # collect()  Returns all the records as a list of Row
 


states.distinct().count()
stations.distinct().count()
countries.distinct().count()
inventory.distinct().count()

states.cache()
stations.cache()
countries.cache()
inventory.cache()
#-------Q3------------------

#(a) The first two characters of the station code denote the country code (FIPS).
stations=stations.withColumn('FIPS',F.substring(F.col('ID'), 1, 2))
stations.show(5)
#(b)LEFT JOIN stations with countries using your output from part (a).
stations=(
    stations
    .join(countries, stations.FIPS == countries.CODE,how="left").drop('CODE')
    .withColumnRenamed('NAME', 'COUNTRY_NAME')
)
stations.count()
#115081

#(c) LEFT JOIN stations and states, allowing for the fact that state codes are only provided for stations in the US.

# List of U.S. state and Canadian Province codes 

#way1
stations_us=(
    stations
    .filter(F.col("FIPS")=="US")
    .join(states, stations.STATE_CODE == states.CODE,how="inner")
    .drop('CODE')
    .select(F.col("ID"),F.col("STATE_CODE"))
) 
#stations_us.count()  #61867

#three jobs:  load 

stations=(
    stations
    .join(stations_us,on="ID",how="left")
    .drop('CODE')
    .withColumnRenamed('NAME', 'STATE_NAME')
) 
stations.count()#115081  
#3 shuffles   

stations.show(5,False)

#way2
stations=(
    stations
    .join(states, stations.STATE_CODE == states.CODE,how="left")
    .drop('CODE')
    .withColumnRenamed('NAME', 'STATE_NAME')
) 

#state codes are only provided for stations in the US. 
stations = (stations
.withColumn("STATE_CODE1", F.when(stations["FIPS"] == "US", stations["STATE_CODE"]).otherwise(""))
.drop('STATE_CODE')
.withColumnRenamed('STATE_CODE1', 'STATE_CODE')
)
stations.count() #115081
# 1 shuffle 

#check the US states
stations.filter(F.col('FIPS')=="US").count()  #61867
stations.filter(F.col("STATE_CODE")!="").show(5)
stations.filter(F.col('FIPS')=="US").show(5)
stations.count()

# (d)

#analysize on inventory
inventory.show(10,False)

#stations 115081      inventory  687141
#check 
inventory.filter((F.col('LASTYEAR')=="") | (F.col('FIRSTYEAR')=="")).count()  #0
inventory.filter((F.col('LASTYEAR').isNotNull()) & (F.col('FIRSTYEAR').isNotNull()) & (F.col('ELEMENT').isNotNull()) ).count()   #687141


from datetime import datetime
import time 
#--d1 what was the first and last year that each station was active and collected any element at all? 
stations_active = (
    inventory
    .groupby('ID')
    .agg(F.min('FIRSTYEAR').alias('OPEN'), F.max('LASTYEAR').alias('CLOSE'))
 ) 
stations_active.show()
stations_active.count()  #115072
inventory.select(F.col('ID')).distinct().count()  #115072

#there are 115081 stations  but only  115072 have element records in  inventory

#--d2  How many different elements has each station collected overall?   

inventory.select(F.col('ELEMENT')).distinct().count()  #137

core_elements_list = ['PRCP', 'SNOW', 'SNWD', 'TMAX', 'TMIN']
# type(core_elements_list)   list

#--#--d2 d3- --way1--using collect_set  and udf  
start_0=datetime.now() 

stations_elementSet=(
 inventory
 .groupby('ID')
 .agg(F.collect_set('ELEMENT'))  #Aggregate function: returns a set of objects with duplicate elements eliminated.
 ) 
#stations_elementSet.show(5,False)
#stations_elementSet.count()  #115072 

# ID  |collect_set(ELEMENT)                  
 
def interElement(conList, allList):  
 #intersection = list(set(conList).intersection(allList)) 
 intersection = list(set(conList) & set(allList))
 return len(intersection)

interElement_udf = F.udf(interElement, IntegerType())

#difference = list(set(b).difference(set(a)))    # elements in b but not in a
def diffElement(conList, allList):
 difference = list(set(allList).difference(set(conList)))
 return len(difference)

diffElement_udf = F.udf(diffElement, IntegerType())

 
stations_elements=(stations_elementSet
 .withColumn("core_elements_list", F.array([F.lit(x) for x in core_elements_list]))   #this is  important 
 .withColumn('ELEMENTS', F.size("collect_set(ELEMENT)"))
 .withColumn('CORE_ELEMENTS',  interElement_udf("core_elements_list", "collect_set(ELEMENT)"))
 .withColumn('OTHER_ELEMENTS', diffElement_udf("core_elements_list", "collect_set(ELEMENT)"))
 )

#stations_elements.count()
stations_elements.show(5)
 
end_0=datetime.now() 
durn_0 = (end_0-start_0).seconds  # unit-second
print(durn_0)   #3
 
#--d2 d3--way2---using filter and join
start_1=datetime.now() 

stations_elements= (
    inventory
    .select(F.col('ID'),F.col('ELEMENT'))
    .distinct()  #make sure the <station and element>  not duplicate   
    .groupby('ID')
    .agg(F.count('ELEMENT').alias('ELEMENTS'))    
 )
#stations_elements.show()
#stations_elements.count()  #115072
 
#count separately the number of core elements and the number of  other  elements that each station has collected overall

inventory_with_coreelements = inventory.filter(F.col('ELEMENT').isin(core_elements_list))  #318583
#filter first
#then 
stations_core_elements= (
    inventory_with_coreelements
    .select(F.col('ID'),F.col('ELEMENT'))
    .distinct()  #make sure the  <station ,element>  are not duplicate   
    .groupby('ID')
    .agg(F.count('ELEMENT').alias('CORE_ELEMENTS'))
 )

#stations_core_elements.count()  #115024

stations_elements=(
      stations_elements
     .join(stations_core_elements,"ID","left") #stations_elements has more stations
     .withColumn('OTHER_ELEMENTS', F.col('ELEMENTS')-F.col('CORE_ELEMENTS'))
)      
 #stations_elements 115072
stations_elements.show(5)
stations_elements.count()
 
end_1=datetime.now() 
durn_1 = (end_1-start_1).seconds  # unit-second
print(durn_1)  #4

 
 
 #--d4 How many stations collect all five core elements? How many only collected temperature
 #TMAX = Maximum temperature (tenths of degrees C)
 #TMIN = Minimum temperature (tenths of degrees C) 
 
stations_elements.sort('CORE_ELEMENTS', ascending=False).show(5)
# As we remove the duplicate records so the max CORE_ELEMENTS is 5

#stations collect all five core elements
stations_elements.filter(F.col('CORE_ELEMENTS')==5).count()  #20266   
stations_elements.filter(F.col('CORE_ELEMENTS')==5).select(F.col("ID")).distinct().count()  #20266   

 #'TMAX', 'TMIN' 
# stations collected temperature
stations_temp_elements_only=(
 stations_elements.filter(F.col('ELEMENTS')<=2)
.join((inventory_with_coreelements.filter((F.col('ELEMENT')=='TMAX') | (F.col('ELEMENT')=='TMIN'))), on='ID', how='inner')
)
stations_temp_elements_only.count()  #614  
stations_temp_elements_only.select(F.col("ID")).distinct().count()   #314
  

# Q3-(e)  join stations with output from (d)

stations_enriched=(
 stations
 .join(stations_elements, "ID","left")  
 .join(stations_active, "ID","left") 
 .withColumnRenamed('ID', 'STATION_ID')
 )   #115018

stations_enriched.show(5)  

#find out the nine stations without elements
stations_enriched.filter(F.col('ELEMENTS').isNull()).select(F.col('STATION_ID'),F.col('ELEMENTS')).dropDuplicates().show()

#/user/xzh216/Assign1/output/
outputpath = "hdfs:///user/xzh216/Assign1/output/"
stations_enriched.write.format('parquet').mode("overwrite").save(outputpath+'stations_enriched.parquet')
stations_enriched.write.format('csv').mode("overwrite").save(outputpath+'stations_enriched.csv')
#stations_enriched.write.csv(outputpath+'stations_enriched.csv',mode="overwrite")   #the same as above
stations_enriched.write.format('orc').mode("overwrite").save(outputpath+'stations_enriched.orc')
 

 
 # Q3-(f)
 
# daily  stations_enriched
stations_enriched = spark.read.orc("hdfs:////user/xzh216/Assign1/output/stations_enriched.orc")
 
stations_enriched.cache()
starttime=datetime.now()   
tempdef=(
    daily
    .join(stations_enriched, daily.ID==stations_enriched.STATION_ID,how="left")
) 
tempdef.show(5)
endtime=datetime.now()  

durn = (endtime-starttime).seconds  # unit-second
print(durn)   # 3seconds


#Are there any stations in your subset of daily that are not in stations at all? 
(tempdef
 .where((F.col("STATION_ID")=="")| (F.col("STATION_ID").isNull()))
 .select(F.col("ID"))
 .dropDuplicates()  #find the  stations not occured in subset of daily 
 ).count() 
#0  there is no stations in the subset of daily that are not in stations at all 

(tempdef
 .select(F.col("ID"))
 .dropDuplicates()   
 ).count()    #318 stations in subset of daily 

(stations_enriched
 .select(F.col("STATION_ID"))
 .dropDuplicates()   
 ).count()  #115081
 
stations_enriched.rdd.getNumPartitions()   #4 

 #---do not using  join    using dropDuplicates()  and  exceptAll
        
(daily
 .select(F.col("ID"))
 .dropDuplicates() 
  .exceptAll(
   (stations_enriched
     .select(F.col("STATION_ID"))
    .dropDuplicates()
    )
  )   
).count()
# the same as above   but using subtract
(daily
 .select(F.col("ID"))
  .subtract(
   (stations_enriched
     .select(F.col("STATION_ID"))
    .dropDuplicates()
    )
  )   
).count()