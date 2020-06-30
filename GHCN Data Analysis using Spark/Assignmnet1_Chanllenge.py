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


path = "hdfs:///data/ghcnd/daily/*.csv.gz"
daily= (
    spark.read.format("com.databricks.spark.csv")
    .option("header", "false")
    .option("inferSchema", "false")
    .schema(schema_Daily)
    .load(path)
)
stations = spark.read.orc("hdfs:////user/xzh216/Assign1/output/stations_enriched.orc")
stations.cache().count()
 

stations.show(5,False)
daily.show(5,False)

#THERE 15 types of values of QUALITY_FLAG
daily.select(F.col('QUALITY_FLAG')).distinct().count()  #15

#find the observations with good quality
daily_good_quality=(
 daily 
 .filter((F.col('QUALITY_FLAG')=="")|(F.col('QUALITY_FLAG').isNull()))
 )
 
 
daily_good_quality.count()  #2919648731

(
 daily 
 .filter(F.col('QUALITY_FLAG')=="")
 ).count()  #0 
 
(
 daily 
 .filter(F.col('QUALITY_FLAG').isNull())
 ).count()   #2919648731
  
 

daily_bad_quality=(
 daily 
 .filter(F.col('QUALITY_FLAG').isNotNull())
 )
daily_bad_quality.persist().count()


(daily_bad_quality
   .groupby('QUALITY_FLAG')
   .agg(F.count("ID").alias('obs_bad_count'))
   .orderBy(F.col("obs_bad_count").desc())
).show(20,False)



 #the quality of the specific observations that each station has collected. 
 #Investigate the coverage of the other elements, both temporally and spatially. 
 
 #How many good observations are there for each stations?
 
#Good_quality=(daily_good_quality
#   .groupby('ID')
#   .agg(F.count("QUALITY_FLAG").alias('obs_good_count'))
#   .orderBy(F.col("obs_good_count").desc())
#)   #obs_good_count is 0 or null

Good_quality=(daily_good_quality
   .groupby('ID')
   .agg(F.count("ID").alias('obs_good_count'))
   .orderBy(F.col("obs_good_count").desc())
)

Good_quality.show()
Good_quality.count()   #115072

 
All_quality=(daily
   .groupby('ID')
   .agg(F.count("ID").alias('obs_all_count'))
   .orderBy(F.col("obs_all_count").desc())
)

All_quality.show(5,False) 
daily.select(F.col('ID')).distinct().count() #115073    #only 115073 stations in daily  
All_quality.count()     # 115073

#there is a station  without good observations

stations.select(F.col('STATION_ID')).distinct().count()  #115081

#there is no stations in daily that are not in stations at all   But there are 8 stations not in daily 
dd=daily.select(F.col('ID')).distinct()
ds=stations.select(F.col('STATION_ID')).distinct()
(dd
 .join(ds, dd.ID == ds.STATION_ID,how="right")
 .where((F.col("ID")=="")| (F.col("ID").isNull()))
 .dropDuplicates()   
 ).show() 

 
 # keep on anaysize the QUALITY_FLAG
Evalations_quality=(All_quality 
  .join(Good_quality, on="ID",how="left") 
  .withColumn('obs_bad_count',F.col('obs_all_count')-F.col('obs_good_count'))
  .withColumn('good_perc',F.col('obs_good_count')/F.col('obs_all_count'))
  .withColumn('bad_perc',F.col('obs_bad_count')/F.col('obs_all_count'))
  .orderBy(F.col("good_perc").desc())
 )

Evalations_quality.persist().count()

# find the top 10 stations with high percentage of quality
Evalations_quality.show(10,False)

# find the only station withno good quality----USC00144840
Evalations_quality.where((F.col("obs_good_count")=="")| (F.col("obs_good_count").isNull())).show()

#how many stations  has good_perc=1
Evalations_quality.filter((F.col("good_perc")==1)).count()   #49185


#how many rows that the 49185 stations overed?
Evalations_quality.filter((F.col("good_perc")==1)).agg(F.sum("obs_good_count")).show()

 
 
 
 
# 
Evalations_quality_countries=(
      Evalations_quality
     .join(stations, daily.ID == stations.STATION_ID,how='left')
     .drop("STATION_ID")
     .groupby('COUNTRY_NAME')
     .agg(F.sum("obs_good_count").alias('good_counts'),
          F.sum("obs_all_count").alias('all_counts'))
     .withColumn('good_perc',F.col('good_counts')/F.col('all_counts'))
     .orderBy(F.col("good_perc").desc())    
 ) 
 
Evalations_quality_countries.show(300,False)

#recheck
Evalations_quality_countries.agg(F.sum("all_counts")).show()   #2928664523
Evalations_quality_countries.agg(F.sum("good_counts")).show()  #2919648731 

 
 
 # foucing on time 
 
Good_quality_year=(daily_good_quality
   .withColumn('YEAR', F.trim(F.substring(F.col('DATE'), 1, 4)).cast(StringType())) 
   .groupby('YEAR')
   .agg(F.count("ID").alias('obs_good_count'))
   .orderBy(F.col("obs_good_count").desc())
)
All_quality_year=(daily
   .withColumn('YEAR', F.trim(F.substring(F.col('DATE'), 1, 4)).cast(StringType())) 
   .groupby('YEAR')
   .agg(F.count("ID").alias('obs_all_count'))
   .orderBy(F.col("obs_all_count").desc())
) 
 
 
Evalations_quality_year=(All_quality_year 
  .join(Good_quality_year, on="YEAR",how="left") 
  .withColumn('obs_bad_count',F.col('obs_all_count')-F.col('obs_good_count'))
  .withColumn('good_perc',F.col('obs_good_count')/F.col('obs_all_count'))
  .withColumn('bad_perc',F.col('obs_bad_count')/F.col('obs_all_count'))
  .orderBy(F.col("YEAR"))
 ) 

Evalations_quality_year.show(258,False)
import pandas
Evalations_quality_year.toPandas().to_csv('/users/home/xzh216/Assign1/Result/Evalations_quality_year.csv')