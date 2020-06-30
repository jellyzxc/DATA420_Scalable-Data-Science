#start_pyspark_shell -e 4 -c 2 -w 4 -m 4 
# Python and pyspark modules required
import sys
from pyspark import SparkContext
from pyspark.sql import SparkSession
from pyspark.sql.types import * 
from pyspark.sql import functions as F
 
 
spark = SparkSession.builder.getOrCreate()
sc = SparkContext.getOrCreate()
# Compute suitable number of partitions

conf = sc.getConf()

N = int(conf.get("spark.executor.instances"))
M = int(conf.get("spark.executor.cores"))
partitions = 3 * N * M

#####################################################################################################################  
#Q2-(a) Filter the Taste Profile dataset to remove the songs which were mismatched

#/data/msd/tasteprofile/mismatches/sid_matches_manually_accepted.txt
#/data/msd/tasteprofile/mismatches/sid_mismatches.txt

 


#1 load the raw sid_mismatches
sid_mismatches = (
     spark.read.format("text")
    .option("delimiter", " ")
    .load("hdfs:///data/msd/tasteprofile/mismatches/sid_mismatches.txt")
)
sid_mismatches.show(5, False)
sid_mismatches.count()  #19094  
# hdfs dfs -cat /data/msd/tasteprofile/mismatches/sid_mismatches.txt | wc -l     #19094  

#get the mismatched  pair  SONG_ID TRACK_ID       
mismatched= (
    sid_mismatches.select(
    F.trim(F.substring(F.col('value'), 9, 18)).alias('SONG_ID'),
    F.trim(F.substring(F.col('value'), 28,18)).alias('TRACK_ID')
    ).orderBy(F.col("SONG_ID").desc())
    .dropDuplicates()
)
mismatched.show(5)
mismatched.count() # 18972
 
 
 
 
#2 load the raw  sid_matches_manually_accepted
sid_matches_manually_accepted = (
    spark.read.format("text")
    .option("delimiter", " ")
    .load("hdfs:///data/msd/tasteprofile/mismatches/sid_matches_manually_accepted.txt")
)
sid_matches_manually_accepted.show(5, False)
sid_matches_manually_accepted.count()  #938

#get the manually accepted  pair  SONG_ID TRACK_ID 
manuallyaccepted = (
    sid_matches_manually_accepted.select
    (
     F.trim(F.substring(F.col('value'), 11, 18)).alias('SONG_ID'),
     F.trim(F.substring(F.col('value'),30,18)).alias('TRACK_ID')
    ).orderBy(F.col("SONG_ID").desc())
    .dropDuplicates()
)
manuallyaccepted.show(5, False)
manuallyaccepted.count() # 526
 

  
#3 add the manually_accepted  back   
truemismatched = mismatched.join(manuallyaccepted, on="SONG_ID", how="left_anti")
truemismatched.count() #18971  
truemismatched.show(5, False)


#save it for    in  Q2  and  Q3
outputpath = "hdfs:///user/xzh216/Assign2/output/"
truemismatched.write.format('orc').mode("overwrite").save(outputpath+'truemismatched.orc')

#get the truely mismatched SongID
truemismatchedSongs=truemismatched.select("SONG_ID").dropDuplicates()
truemismatchedSongs.count()  #18912
 
 
#Taste Profile:  the official user dataset of the Million Song Datase
#The dataset contains real user - play counts from undisclosed partners, all songs already matched to the MSD
#4 load the Taste Profile dataset    (user, song, play count) triplets    
schema_taste = StructType([
    StructField('USER_ID', StringType()),
    StructField('SONG_ID', StringType()),
    StructField('PLAY_COUNT', IntegerType()),
])

taste_raw = (
    spark.read.format("com.databricks.spark.csv")
    .option("header", "false")
    .option("inferSchema", "false")
    .option("delimiter", "\t")
    .schema(schema_taste)
    .load("hdfs:///data/msd/tasteprofile/triplets.tsv/*.*")
)
taste_raw.rdd.getNumPartitions()   #8 artitions
taste_raw.show(10, False)
#+----------------------------------------+------------------+----------+
#|USER_ID                                 |SONG_ID           |PLAY_COUNT|
#+----------------------------------------+------------------+----------+
#|f1bfc2a4597a3642f232e7a4e5d5ab2a99cf80e5|SOQEFDN12AB017C52B|1         |
  
taste_raw.count() # 48373586
# hdfs dfs -cat /data/msd/tasteprofile/triplets.tsv/part-* | gunzip | wc -l   #48373586      

#1,019,318 unique users
#384,546 unique MSD songs
#48,373,586 user - song - play count triplets
 
 
#5 remove truely mismatched Songs
# Left Anti Join  This join is like df1-df2, as it selects all rows from df1 that are not present in df2. 
taste = taste_raw.join(truemismatchedSongs, how='left_anti', on='SONG_ID')
taste.count() # 45795111     

outputpath = "hdfs:///user/xzh216/Assign2/output/"
taste.write.format('orc').mode("overwrite").save(outputpath+'taste.orc')
 
 
 
 
 
####========taste :  the clean  user - play counts from undisclosed partners


########################################################################################################################### 
#Q2-(b)Load the audio feature attribute names and types from the audio/attributes directory  
#      and use them to define schemas for the audio features themselves.

  
# 1  check all the unique datatype of attritbute: string, STRING, NUMERIC, real
#_c0    _c1   
attributes_all = (
    spark.read.format("com.databricks.spark.csv")
    .load("hdfs:///data/msd/audio/attributes/*.*")
    .dropDuplicates()
)
attributes_all.show(100, False)

attributes_datatype = (
     attributes_all  
    .select('_c1')
    .dropDuplicates()
)
attributes_datatype.show(4, False)
attributes_datatype.count()  #4
#+-------+
#|_c1    |
#+-------+
#|string |
#|STRING |
#|NUMERIC|
#|real   |
#+-------+



# 2 define schemas for the audio features themselves. 
  
#the attribute files and feature datasets share the same prefix and that the attribute types are named  consistently

# feature_fullName_list = [ Rhythm_Patterns,Statistical_Spectrum_Descriptors,Rhythm_Histograms,Temporal_Statistical_Spectrum_Descriptors,	 	
                # Temporal_Rhythm_Histograms,MVD,MARSYAS_timbral_features,Low-level_features,Low-level_features_derivatives,
                # Method_of_Moments,Area_of_Moments,Linear_Predictive_Coding,MFCC_features]
              
featureSet_list = ["RP","SSD","RH","TSSD","TRH","MVD","MT","JS","JSD","JMM","JAM","PLC","MFCC"] 
prefix_list = ["RP_","SSD_","RH_","TSSD_","TRH_","MVD_","MT_","JS_","JSD_","JMM_","JAM_","PLC_","MFCC_"]               
file_list=[
    "msd-rp-v1.0",
    "msd-ssd-v1.0",
    "msd-rh-v1.0",
    "msd-tssd-v1.0",
    "msd-trh-v1.0", 
    "msd-mvd-v1.0",
    "msd-marsyas-timbral-v1.0",
    "msd-jmir-spectral-all-all-v1.0",
    "msd-jmir-spectral-derivatives-all-all-v1.0", 
    "msd-jmir-methods-of-moments-all-v1.0",
    "msd-jmir-area-of-moments-all-v1.0",
    "msd-jmir-lpc-all-v1.0",
    "msd-jmir-mfcc-all-v1.0" 
] 
 
# build a dictionary mapping attributes types to pysprak.sql.type objects  
type_map = {
  "NUMERIC": DoubleType(),
  "real": DoubleType(),
  "string": StringType(),
  "STRING": StringType()
}

#generate featureSet_schemas 
feature_schema_list = {}
for index in range(0,13):

    #generate schema
    filename=file_list[index]
    print(filename)
    attr_path = "hdfs:///data/msd/audio/attributes/" + filename + '.attributes.csv'
    attributes = spark.read.format("com.databricks.spark.csv").load(attr_path)
    rows=attributes.collect()
    prefix=prefix_list[index]
    feature_schema= StructType([StructField(prefix+row['_c0'], type_map[row['_c1']], True) for row in rows])   # difficult    add an prefix    
    
    #load featureset data
    feature_path = "hdfs:///data/msd/audio/features/" + filename + '.csv/*.*'
    df = (
        spark.read.format("com.databricks.spark.csv")
        .schema(feature_schema)
        .load(feature_path)
        )
    #save data
    locals()[featureSet_list[index]]=df 
  
 
#test
RH.show(5,False)
RH.printSchema()
 
for index in range(0,13):
    attr_path = "hdfs:///data/msd/audio/attributes/" + file_list[index] + '.attributes.csv'
    df = (
        spark.read.format("com.databricks.spark.csv")
        .load(attr_path)
        )
    print(file_list[index])
    print(featureSet_list[index],df.count(),sep=":") 
#====Attribute Counts
# msd-rp-v1.0
# RP:1441
# msd-ssd-v1.0
# SSD:169
# msd-rh-v1.0
# RH:61
# msd-tssd-v1.0
# TSSD:1177
# msd-trh-v1.0
# TRH:421
# msd-mvd-v1.0
# MVD:421
# msd-marsyas-timbral-v1.0
# MT:125
# msd-jmir-spectral-all-all-v1.0
# JS:17
# msd-jmir-spectral-derivatives-all-all-v1.0
# JSD:17
# msd-jmir-methods-of-moments-all-v1.0
# JMM:11
# msd-jmir-area-of-moments-all-v1.0
# JAM:21
# msd-jmir-lpc-all-v1.0
# PLC:21
# msd-jmir-mfcc-all-v1.0
# MFCC:27

 
 
 for index in range(0,13): 
       print(file_list[index]) 
       print(featureSet_list[index],locals()[featureSet_list[index]].count(),sep=":") 
#====FeatureSet Counts
# msd-rp-v1.0
# RP:994188
# msd-ssd-v1.0
# SSD:994188
# msd-rh-v1.0
# RH:994188
# msd-tssd-v1.0
# TSSD:994188
# msd-trh-v1.0
# TRH:994188
# msd-mvd-v1.0
# MVD:994188
# msd-marsyas-timbral-v1.0
# MT:995001
# msd-jmir-spectral-all-all-v1.0
# JS:994623
# msd-jmir-spectral-derivatives-all-all-v1.0
# JSD:994623      #the  same  as  hdfs  994623
# msd-jmir-methods-of-moments-all-v1.0
# JMM:994623
# msd-jmir-area-of-moments-all-v1.0
# JAM:994623
# msd-jmir-lpc-all-v1.0
# PLC:994623
# msd-jmir-mfcc-all-v1.0
# MFCC:994623

  
#double check
MFCC.count()
JS.count()
 
 
  
featureSets= [RP,SSD,RH,TSSD,TRH,MVD,MT,JS,JSD,JMM,JAM,PLC,MFCC] 
for feature_df in featureSets:
    feature_df.printSchema()
 
for feature_df in featureSets:
    feature_df.show(2,False)
    
# chang the last column name to 'TRACK_ID'  
# remove the '' from the track_ID column
 
for index in range(0,13):
    feature_df=featureSets[index]
    df=feature_df.withColumnRenamed(feature_df.columns[-1], 'TRACK_ID')
    track_ID_vaule_0 = df.select(F.trim(F.col('TRACK_ID'))).rdd.map(lambda x: x[0]).take(1)[0]
    if track_ID_vaule_0.startswith("'"):
        df = df.withColumn(
            'TRACK_ID', F.substring(F.trim(F.col('TRACK_ID')),2,18)   #'TRKXUVI12903CF164C'|
            )   
    locals()[featureSet_list[index]]=df
    df.write.format('orc').mode("overwrite").save(outputpath+featureSet_list[index]+'.orc')
    
featureSets= [RP,SSD,RH,TSSD,TRH,MVD,MT,JS,JSD,JMM,JAM,PLC,MFCC]
 
 #==== 13 clean feature sets  in featureSets
for index in range(0,13):
    df=featureSets[index]
    print( file_list[index], featureSet_list[index], len(df.columns), df.count(),sep=":")
 
# msd-rp-v1.0:RP:1441:994188
# msd-ssd-v1.0:SSD:169:994188
# msd-rh-v1.0:RH:61:994188
# msd-tssd-v1.0:TSSD:1177:994188
# msd-trh-v1.0:TRH:421:994188
# msd-mvd-v1.0:MVD:421:994188
# msd-marsyas-timbral-v1.0:MT:125:995001
# msd-jmir-spectral-all-all-v1.0:JS:17:994623
# msd-jmir-spectral-derivatives-all-all-v1.0:JSD:17:994623
# msd-jmir-methods-of-moments-all-v1.0:JMM:11:994623
# msd-jmir-area-of-moments-all-v1.0:JAM:21:994623
# msd-jmir-lpc-all-v1.0:PLC:21:994623
# msd-jmir-mfcc-all-v1.0:MFCC:27:994623


