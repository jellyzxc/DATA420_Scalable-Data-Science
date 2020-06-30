import sys
from pyspark import SparkContext
from pyspark.sql import SparkSession
from pyspark.sql.types import * 
from pyspark.sql import functions as F

spark = SparkSession.builder.getOrCreate()
sc = SparkContext.getOrCreate()
#================== Q3 ====================--
 
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

daily_2020 = (
    spark.read.format("com.databricks.spark.csv")
    .option("header", "false")
    .option("inferSchema", "false")
    .schema(schema_Daily)
    .load("hdfs:///data/ghcnd/daily/2020.csv.gz")
)
daily_2020.count()  #5215365   ---2 stage 1 task each stage

daily_2020.show(5)


daily_2015 = (
    spark.read.format("com.databricks.spark.csv")
    .option("header", "false")
    .option("inferSchema", "false")
    .schema(schema_Daily)
    .load("hdfs:///data/ghcnd/daily/2015.csv.gz")
)
#load not tirgger job
daily_2015.count()  #34899014    ---2 stage 1 task each stage   



daily_2015.show(5)  
daily_2020.rdd.getNumPartitions()  #1
daily_2015.rdd.getNumPartitions() #1    33.28M  1 Partition    ---2 stage 1 task each stage    



#"hdfs:///data/ghcnd/daily/20(([1][5-9])|20).csv.gz"    () |   doesnot work 
# Linux wildcard  is not the  same as regular expressions 
path_2015to2020 = "hdfs:///data/ghcnd/daily/20{{[1][5-9]},20}.csv.gz"
daily_2015to2020= (
    spark.read.format("com.databricks.spark.csv")
    .option("header", "false")
    .option("inferSchema", "false")
    .schema(schema_Daily)
    .load(path_2015to2020)
)
daily_2015to2020.count()
# 178918901      170.63M  6 Partitions   --28.4M

daily_2015to2020.rdd.getNumPartitions()  #6 
 
# repartition(numPartitions, *cols)[source]  Returns a new DataFrame partitioned by the given partitioning expressions. 
 

#sc.defaultMinPartitions   #2
#sc.defaultParallelism   #8

# according to TextInputFormat.getSplits   =>numSplits 