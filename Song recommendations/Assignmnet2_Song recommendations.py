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

############################################  Q 1   #########################################################################  

 #taste = taste_raw.join(truemismatchedSongs, how='left_anti', on='SONG_ID')
 #taste.count() # 45795111     
 
#Load the clean taste dataset  saved in processing
taste = spark.read.orc("hdfs:///user/xzh216/Assign2/output/taste.orc") 
taste.show(2,False)

# +------------------+----------------------------------------+----------+
# |SONG_ID           |USER_ID                                 |PLAY_COUNT|
# +------------------+----------------------------------------+----------+
# |SOFQZJF12AB0186382|e125b9c74cc8dd73bfae034cc18739b33b837abe|2         |
# |SOFWBNH12A58A7C8CD|e125b9c74cc8dd73bfae034cc18739b33b837abe|1         |
# +------------------+----------------------------------------+----------+
 
#======= Q1 a 
Song_Count=taste.select(F.col("SONG_ID")).distinct().count()  #378310
User_Count=taste.select(F.col("USER_ID")).distinct().count()   # 1019318

taste.distinct().count()
taste.rdd.getNumPartitions()
taste = taste.repartition(partitions).cache()
#======= Q1 b
# How many different songs has the most active user played?  #4316          which is almost 3 times  then the  second one            
# What is this as a percentage of the total number of unique songs in the dataset?   #0.011408   1.14%

#the most active user: user who play the most unique songs
#for the user, how many unique songs they have played  and   how many times they played ?

from pyspark.sql import Window   
user_active = (taste
         .groupBy('USER_ID')  
         .agg(F.count('SONG_ID').alias('SongCounts'), 
              F.sum('PLAY_COUNT').alias('Sum_PlayCounts_U'))
          .withColumn('Percent', F.col('SongCounts')/Song_Count )
          .orderBy('SongCounts', ascending = False) 
          ) 
windowSpec = Window.orderBy(F.desc('SongCounts'))           
user_active =(user_active.withColumn('cumsum_Percentage', F.sum(user_active['Percent']).over(windowSpec)))       
user_active.show(5,False) 
 
# +----------------------------------------+----------+----------------+--------------------+--------------------+
# |USER_ID                                 |SongCounts|Sum_PlayCounts_U|Percent             |cumsum_Percentage   |
# +----------------------------------------+----------+----------------+--------------------+--------------------+
# |ec6dfcf19485cb011e0b22637075037aae34cf26|4316      |5146            |0.011408633131558774|0.011408633131558774|
# |8cb51abc6bf8ea29341cb070fe1e1af5e4c3ffcc|1562      |2599            |0.004128889006370437|0.015537522137929211|
# |5a3417a1955d9136413e0d293cd36497f5e00238|1557      |1679            |0.004115672332214322|0.019653194470143534|
# |fef771ab021c200187a419f5e55311390f850a50|1545      |2847            |0.004083952314239645|0.023737146784383177|
# |c1255748c06ee3f6440c51c439446886c7807095|1498      |4977            |0.003959715577172161|0.027696862361555337|
# +----------------------------------------+----------+----------------+--------------------+--------------------+

#------------check    the max cumsum_Percentage>>1  as different  users will listen the same song   
user_active.orderBy("cumsum_Percentage",ascending = False).show(5,25)
#============================================================================================================================== 
#=#=#=#=        cumsum is not correct here as users will listen the same song and the sum(Percent)will larger then 1   #=#=#=#=     
#=============================================================================================================================== 
user_active.orderBy("SongCounts",ascending = True).show(5,25)


threshold=50
robot_users=taste.filter(F.col("PLAY_COUNT")>threshold).select(F.col("USER_ID")).distinct() 
robot_users.count() #55507


# user_active=user_active.withColumn('RepeatRate', (F.col('Sum_PlayCounts_U')-F.col('SongCounts'))/F.col('SongCounts'))
# user_active.orderBy('RepeatRate', ascending = False).show(10)
# threshold=10
# #get possible robot users
# robot_users=user_active.filter(F.col("RepeatRate")>threshold).select(F.col("USER_ID")).distinct()
# robot_users.count()  # 25507

#1019318-55507   963811
 
#it is better to use filter taste.filter(F.col("PLAY_COUNT")>threshold)
taste = taste.join(robot_users, how='left_anti', on='USER_ID').distinct()
taste.cache().count() 
   
#45795111=>42394460

Song_Count=taste.select(F.col("SONG_ID")).distinct().count()  #  378310=>376343 
User_Count=taste.select(F.col("USER_ID")).distinct().count()  # 1019318=>963811
print(Song_Count,User_Count)  # 376343 963811
 
 

  
user_active = (taste
         .groupBy('USER_ID')  
         .agg(F.count('SONG_ID').alias('SongCounts'), 
              F.sum('PLAY_COUNT').alias('Sum_PlayCounts_U'))
          .withColumn('Percent', F.col('SongCounts')/Song_Count )
          .orderBy('SongCounts', ascending = False) 
          )        
user_active.show(5,False) 
 

 
#======= Q1 c===========
# Visualize the distribution of song popularity and the distribution of user activity.
# What is the shape of these distributions?

#for the song, how many unique users played it and how many times it was played
song_popularity = (taste
         .groupBy('SONG_ID')  
         .agg(F.count('USER_ID').alias('UserCounts'), 
              F.sum('PLAY_COUNT').alias('Sum_PlayCounts_1'))
          .withColumn('Percent_1', F.col('UserCounts')/User_Count )
          .orderBy('UserCounts', ascending = False) 
          )
song_popularity.show(5,False)
 
 
 

#-----plot
user_activity_pandas = user_active.toPandas()
song_popularity_pandas = song_popularity.toPandas()

 
import numpy as np
import matplotlib.pyplot as plt

 
 
#The distribution of user activity
y = user_activity_pandas["SongCounts"]
x = np.arange(1, len(y)+1) 
plt.clf()
plt.figure(figsize=(16,9))
plt.plot(x, y, 'g-')
plt.xlabel("users ordered by activity")
plt.ylabel('# of songs which was listend by users ')
plt.title("The distribution of user activity")
plt.savefig('The distribution of user activity .png')

 

# Empirical cumulative distribution function (ECDF) in Python
#https://cmdlinetips.com/2019/05/empirical-cumulative-distribution-function-ecdf-in-python/ 
x = np.sort(user_activity_pandas["SongCounts"])
y = np.arange(1, len(x)+1) / len(x)
plt.clf()
plt.figure(figsize=(16,9))
plt.plot(x, y, 'g-')
plt.xlabel("# of songs listend by users")
plt.ylabel('ECDF')
plt.xlim((0,500))
plt.title('Empirical cumulative distribution of user activity')
plt.savefig('Empirical cumulative distribution of user activity.png')


#========================

#The distribution of song popularity
y = song_popularity_pandas["UserCounts"]
x = np.arange(1, len(y)+1) 
plt.clf()
plt.figure(figsize=(16,9))
plt.plot(x, y, 'g-')
plt.xlabel("song ordered by popularity")
plt.ylabel('# of users who listened the song')
plt.title("The distribution of song popularity ")
plt.savefig('The distribution of song popularity .png')

 
#ECDF
x = np.sort(song_popularity_pandas["UserCounts"])
y = np.arange(1, len(x)+1) / len(x)
plt.clf()
plt.figure(figsize=(16,9))
plt.plot(x, y, 'g-')
plt.xlabel("# of users who listened the song")
plt.ylabel('ECDF')
plt.xlim((0,500))
plt.title('Empirical cumulative distribution of song popularity')
plt.savefig('Empirical cumulative distribution of song popularity.png')

  
#60  70
 
 
#======= Q1 d===========
# Create a clean dataset of user-song plays by removing songs which have been played less
# than N times and users who have listened to fewer than M songs in total. 
# Choose sensible values for N and M and justify your choices, taking into account (a) and (b).

N=60    #songs which have been played lessthan N times      PLAY_COUNT
M=70    #users who have listened to fewer than M songs

 
Unactive_users = user_active.filter(F.col("SongCounts")<M).select(F.col("USER_ID")).distinct()
Unpopular_song = song_popularity.filter(F.col("UserCounts")<N).select(F.col("SONG_ID")).distinct()

 
               
taste_adjust =(taste
                .join(Unactive_users, how='left_anti', on='USER_ID')   #REMOVE  Unactive_users
                .join(Unpopular_song, how='left_anti', on='SONG_ID')   #REMOVE  Unpopular_song
               )
               
               
               
               
taste_adjust.cache()
taste_adjust.show(10,25)
taste_adjust.count()  
#45795111=>42394460=> 21721180

Song_Count_adjust=taste_adjust.select(F.col("SONG_ID")).distinct().count()  #378310=>376343 =>  85913   
User_Count_adjust=taste_adjust.select(F.col("USER_ID")).distinct().count()   # 1019318=>963811=>  179891
print(Song_Count_adjust,User_Count_adjust)  
#85913 179891

 
##======= Q1 e===========
# (e) Split the user-song plays into training and test sets. Make sure that the test set contains at
# least 20% of the plays in total.


# if some users in testing set do not have records in training set, 
# the matrix of user-latent generated by model will have no corresponding vector for this user,
# it cannot make recommendations for them.

from pyspark.ml.feature import VectorAssembler, PCA, StringIndexer, StandardScaler
from pyspark.ml import Pipeline

#encode USER_ID  SONG_ID
label_user = StringIndexer(inputCol = "USER_ID", outputCol = "user")
label_song = StringIndexer(inputCol = "SONG_ID", outputCol = "item")
pipeline = Pipeline(stages=[label_user, label_song])


pipeline_model = pipeline.fit(taste_adjust)
taste_transformed = pipeline_model.transform(taste_adjust)
taste_transformed.show(3, False)

# +------------------+----------------------------------------+----------+-------+------+
# |SONG_ID           |USER_ID                                 |PLAY_COUNT|user   |item  |
# +------------------+----------------------------------------+----------+-------+------+
# |SOAAFYH12A8C13717A|0b8e37b3e6c51dddd6413126c3e6a7487bcc6e92|3         |79398.0|6740.0|
# |SOAAFYH12A8C13717A|30f338cba0d6e3e6c27f1e90952b269341084a66|18        |97423.0|6740.0|
# |SOAAFYH12A8C13717A|3860e99293381182205306e1b7bde618da89bff5|12        |57504.0|6740.0|
# +------------------+----------------------------------------+----------+-------+------+
 

# create a column holding number of rows in the window partitioned by user
windowSpec = Window.partitionBy("USER_ID").orderBy('SONG_ID')
taste_ordered = (taste_transformed.withColumn("row_num",F.row_number().over(windowSpec)))
  
 
taste_ordered.cache()
taste_ordered.count()  #21721180
taste_ordered.select(F.col('USER_ID')).distinct().count()  #179891
taste_ordered.show(2000, 25)
 
 
# split dataset to training(80%) and testing data(20%)
# M=70 users who have listened to fewer than 70 songs  are not  in the dataset  
# howerver ,we also remove songs  which will affect the SongCounts in  taste_adjust 


train = taste_ordered.filter((F.col("row_num")%5)!=0)    
test = taste_ordered.filter((F.col("row_num")%5)==0)
 

print('train.count: ', train.count())  
print('test.count: ', test.count())   
print('unique users counts in training dataset: ',train.select(F.col('USER_ID')).distinct().count()) 
print('unique users counts intesting dataset: ',test.select(F.col('USER_ID')).distinct().count()) 
 
# train.count:  17448754
# test.count:  4272426
# unique users counts in training dataset:  179891
# unique users counts intesting dataset:  179889



#===we still   ensure that every user in the test set has some user-song plays in the training set as well.

userAll=taste_ordered.select(F.col('user')).distinct()
userTR=train.select(F.col('user')).distinct()
userTS=test.select(F.col('user')).distinct()
(userTR.join(userTS, how='left_anti',on='user')).show(10,False)     
 
# +--------+
# |user    |
# +--------+
# |179889.0|
# |179890.0|
 
train.filter((F.col('user')==179889) | (F.col('user')==179890 )).show(10,False)     
# +------------------+----------------------------------------+----------+--------+-------+-------+
# |SONG_ID           |USER_ID                                 |PLAY_COUNT|user    |item   |row_num|
# +------------------+----------------------------------------+----------+--------+-------+-------+
# |SOBRXJY12A58A7C5BB|ea49cf329dc9addca2d3357b4547f6e2ce94f055|1         |179890.0|54425.0|1      |
# |SOONMOB12A8C13D9F4|ea49cf329dc9addca2d3357b4547f6e2ce94f055|2         |179890.0|68353.0|2      |
# |SOUMMMX12A6D4F66D6|ea49cf329dc9addca2d3357b4547f6e2ce94f055|1         |179890.0|14327.0|3      |
# |SOYKQDS12A8AE46AAB|ea49cf329dc9addca2d3357b4547f6e2ce94f055|1         |179890.0|71689.0|4      |
# |SOBWNLS12AC46876E2|cfa09559db84af9ba8ecb40c41bffad4bcfb3e2c|1         |179889.0|40397.0|1      |
# |SOGFSSJ12AC46882F5|cfa09559db84af9ba8ecb40c41bffad4bcfb3e2c|1         |179889.0|41033.0|2      |
# |SOHKPGY12A6D4F707D|cfa09559db84af9ba8ecb40c41bffad4bcfb3e2c|2         |179889.0|80227.0|3      |
# |SOVXGQG12AC46876F2|cfa09559db84af9ba8ecb40c41bffad4bcfb3e2c|1         |179889.0|52860.0|4      |


(taste_adjust
         .filter((F.col('USER_ID')=="ea49cf329dc9addca2d3357b4547f6e2ce94f055") | (F.col('USER_ID')=="cfa09559db84af9ba8ecb40c41bffad4bcfb3e2c" ))
         .groupBy('USER_ID')  
         .agg(F.count('SONG_ID').alias('SongCounts'), 
              F.sum('PLAY_COUNT').alias('Sum_PlayCounts_U'))
          .orderBy('SongCounts', ascending = False) 
 ).show(2,False)
# |USER_ID                                 |SongCounts|Sum_PlayCounts_U|
# +----------------------------------------+----------+----------------+
# |cfa09559db84af9ba8ecb40c41bffad4bcfb3e2c|4         |5               |
# |ea49cf329dc9addca2d3357b4547f6e2ce94f055|4         |5              


# we remove songs using N  which will affect the SongCounts in  taste_adjust 

 
 #save for next day 
outputpath = "hdfs:///user/xzh216/Assign2/output/"
train.write.format('orc').mode("overwrite").save(outputpath+'train.orc')
test.write.format('orc').mode("overwrite").save(outputpath+'test.orc')  
   
 
 ############################################  Q 2   #########################################################################  
 ##======= Q2 a===========
 #Use the spark.ml library to train an implicit matrix factorization model using Alternating Least Squares (ALS
 #Load the clean taste dataset  saved in processing
trainingData = spark.read.orc("hdfs:///user/xzh216/Assign2/output/train.orc") 
testData = spark.read.orc("hdfs:///user/xzh216/Assign2/output/test.orc") 
 
trainingData.cache()
testData.cache()
trainingData.show()
from pyspark.ml.recommendation import ALS
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.mllib.evaluation import RankingMetrics

# treat PLAY_COUNT as rating
trainingData = trainingData.withColumn('rating',F.col('PLAY_COUNT'))
testData = testData.withColumn('rating',F.col('PLAY_COUNT'))


#train an implicit matrix factorization model
als = ALS(maxIter=5, regParam=0.01, userCol="user", itemCol="item", ratingCol="rating",seed=1000,implicitPrefs=True)
alsModel = als.fit(trainingData)
 
#=============predictions from the test set 
predictions = alsModel.transform(testData) #take (user,item)   to  score(rating), that is the prediction
predictions.show(5, False)

# |SONG_ID           |USER_ID                                 |PLAY_COUNT|user  |item|row_num|rating|prediction|
# +------------------+----------------------------------------+----------+------+----+-------+------+----------+
# |SOPUCYA12A8C13A694|1aa4fd215aadb160965110ed8a829745cde319eb|1         |18.0  |12.0|695    |1     |0.5772592 |
# |SOPUCYA12A8C13A694|e8f6a8d06b0096737dec1a9f44c3d48cd9e5e4b8|1         |161.0 |12.0|425    |1     |0.64405406|
# |SOPUCYA12A8C13A694|c7ac56114cefd8af85f3dd5522ac31406f345d2c|1         |633.0 |12.0|335    |1     |0.50107056|
# |SOPUCYA12A8C13A694|a8728523fe8f0a1e8421578e1b39c175b04f59ff|31        |1250.0|12.0|270    |31    |1.1822103 |
# |SOPUCYA12A8C13A694|27a9784c9359f581fc70c9bd1f3a55e4dc177898|6         |1314.0|12.0|270    |6     |0.76983774|

evaluator = RegressionEvaluator(metricName="rmse", labelCol="rating", predictionCol="prediction")
evaluator.evaluate(predictions.filter(F.col('prediction') != np.NaN))  #5.622484223242347
 
 
##======= Q2 b==========
# Select a few of the users from the test set by hand and use the model to generate some
# recommendations. Compare these recommendations to the songs the user has actually
# played. Comment on the effectiveness of the collaborative filtering model.
 
K= 10 #number of item to recommend 
recommendations=alsModel.recommendForAllUsers(K) 
recommendations.show(5,False)
recommendations.cache()

#|user|recommendations    

# ------------------------------------------------------+
# |12  |[[276, 0.92054075], [368, 0.8984277], [299, 0.883101], [107, 0.86487603], [214, 0.8210335], [67, 0.75282973], [62, 0.70897347], [2701, 0.67073584], [1699, 0.6643137], [161, 0.6520203]]|
# |18  |[[10, 0.9460181], [72, 0.91653275], [6, 0.87843245], [195, 0.8635022], [7, 0.85495335], [151, 0.85127455], [421, 0.82878643], [339, 0.7972208], [341, 0.7899889], [11, 0.7790586]]      |
# |38  |[[117, 1.1146648], [0, 1.1076634], [58, 1.0590011], [32, 0.98469275], [134, 0.95175236], [221, 0.9064955], [158, 0.90067476], [64, 0.89783055], [190, 0.87965655], [192, 0.87497216]]   |
# |67  |[[32, 1.342261], [58, 1.3416071], [64, 1.2729886], [158, 1.2687172], [188, 1.1936744], [192, 1.1906801], [117, 1.1679562], [248, 1.0908555], [216, 1.0837134], [190, 1.0788208]]        |
# |70  |[[10, 1.281222], [7, 1.270271], [3, 1.1653211], [103, 1.1415759], [6, 1.1289587], [177, 1.0642041], [11, 1.0262775], [16, 1.000448], [19, 0.9832375], [43, 0.96991783]]                 |
# +----+---

user1=12
user2=199
user3=67676

#I GUESS recommendForAllUsers are based on training data 
trainingData.filter(F.col('user')==user1).select("item","PLAY_COUNT").orderBy('PLAY_COUNT', ascending = False).select("item").show(K,False)
taste_ordered.filter(F.col('user')==user1).select("item","PLAY_COUNT").orderBy('PLAY_COUNT', ascending = False).select("item").show(K,False)
predictions.filter(F.col('user')==user1).select("item","PLAY_COUNT").orderBy('PLAY_COUNT', ascending = False).select("item").show(K,False)
testData.filter(F.col('user')==user1).select("item","PLAY_COUNT").orderBy('PLAY_COUNT', ascending = False).select("item").show(K,False)
 
 
recommendations.filter(F.col('user')==user2).select("recommendations.item").show(1,False)
taste_ordered.filter(F.col('user')==user2).select("item","PLAY_COUNT").orderBy('PLAY_COUNT', ascending = False).select("item").show(K,False)
trainingData.filter(F.col('user')==user2).select("item","PLAY_COUNT").orderBy('PLAY_COUNT', ascending = False).select("item").show(K,False)
predictions.filter(F.col('user')==user2).select("item","PLAY_COUNT").orderBy('PLAY_COUNT', ascending = False).select("item").show(K,False)
testData.filter(F.col('user')==user2).select("item","PLAY_COUNT").orderBy('PLAY_COUNT', ascending = False).select("item").show(K,False)
 
 
 
recommendations.filter(F.col('user')==user3).select("recommendations.item").show(1,False)
taste_ordered.filter(F.col('user')==user3).select("item","PLAY_COUNT").orderBy('PLAY_COUNT', ascending = False).select("item").show(K,False)
trainingData.filter(F.col('user')==user3).select("item","PLAY_COUNT").orderBy('PLAY_COUNT', ascending = False).select("item").show(K,False)
predictions.filter(F.col('user')==user3).select("item","PLAY_COUNT").orderBy('PLAY_COUNT', ascending = False).select("item").show(K,False)
testData.filter(F.col('user')==user3).select("item","PLAY_COUNT").orderBy('PLAY_COUNT', ascending = False).select("item").show(K,False)
 
 
# +-----------------------------------------------------+
# |item                                                 |
# +-----------------------------------------------------+
# |[214, 276, 107, 299, 67, 2701, 368, 2658, 2213, 1699]|
# +-----------------------------------------------------+

# +-------+
# |item   |
# +-------+
# |2240.0 |
# |5075.0 |
# |4946.0 |
# |19038.0|
# |34973.0|
# |3084.0 |
# |19165.0|
# |27021.0|
# |26649.0|
# |214.0  |
# +-------+
# only showing top 10 rows

# +-------+
# |item   |
# +-------+
# |2240.0 |
# |4946.0 |
# |19165.0|
# |34973.0|
# |26649.0|
# |7026.0 |
# |27021.0|
# |214.0  |
# |2114.0 |
# |22982.0|
# +-------+
# only showing top 10 rows

# +-------+
# |item   |
# +-------+
# |5075.0 |
# |19038.0|
# |3084.0 |
# |26318.0|
# |5622.0 |
# |37107.0|
# |20354.0|
# |32027.0|
# |4778.0 |
# |14788.0|
# +-------+
# only showing top 10 rows

# +-------+
# |item   |
# +-------+
# |5075.0 |
# |3084.0 |
# |19038.0|
# |5622.0 |
# |26318.0|
# |66623.0|
# |4587.0 |
# |14788.0|
# |41408.0|
# |3335.0 |
 
 
 
##======= Q2 c==========
#compute the following metrics
 
 

windowSpec = Window.partitionBy('user').orderBy(F.col('prediction').desc())
song_recommendation = (
     predictions 
    .select('user', 'item', 'prediction', F.rank().over(windowSpec).alias('rank'))  
    .where(F.col('rank')<=K)  
    .groupBy('user')  
    .agg(F.collect_list('item').alias('song_recommended'))
    )
song_recommendation.cache()
song_recommendation.show(10,False)

# from test dataset, collect all the songs a users acutally listend
# into a array in a single columns, begin with the one with highest rating
windowSpec = Window.partitionBy('user').orderBy(F.col('rating').desc())
song_actuallyplayed =(
     testData  
    .select('user', 'rating', 'item', F.rank().over(windowSpec).alias('rank'))  
    .where(F.col('rank')<=K)  
    .groupBy('user')  
    .agg(F.collect_list('item').alias('song_played'))
    )
song_actuallyplayed.cache()
song_actuallyplayed.show(5,False)
 
recommendation_and_played= song_recommendation.join(song_actuallyplayed, on='user') 
recommendation_and_played.cache()
recommendation_and_played.show(5, False)

#predictionAndLabels 
 
metrics = RankingMetrics(
    recommendation_and_played  
    .select(F.col('song_recommended'), F.col('song_played'))  
    .rdd  
    .map(lambda row: (row[0], row[1]))
    )

print("PRECISION @ 5: ", metrics.precisionAt(5))
print("NDCG @10: ", metrics.ndcgAt(10))
print("MAP: ", metrics.meanAveragePrecision)
# PRECISION @ 5:  0.7800820267978149
# NDCG @10:  0.7721561039457752
# MAP:  0.3971339424568883


 
#=============recommendForAllUsers  ==================
# recommendForAllUsers(numItems)[source]¶
# Returns top numItems items recommended for each user, for all users.
# Parameters
# numItems – max number of recommendations for each user
# Returns
# a DataFrame of (userCol, recommendations), where recommendations are stored as an array of (itemCol, rating) Rows.

song_recommendation2=recommendations.select("user","recommendations.item") 
song_recommendation2.show(5,False)
recommendation_and_played_2= song_recommendation2.join(song_actuallyplayed, on='user')
recommendation_and_played_2.show(5,False) 

metrics2 = RankingMetrics(
    recommendation_and_played_2  
    .select(F.col('item'), F.col('song_played'))  
    .rdd  
    .map(lambda row: (row[0], row[1]))
    )

print("PRECISION @ 5: ", metrics2.precisionAt(5))
print("NDCG @10: ", metrics2.ndcgAt(10))
print("MAP: ", metrics2.meanAveragePrecision)
# PRECISION @ 5:  0.04033042598491294
# NDCG @10:  0.03822780482636813
# MAP:  0.009153028688669382
