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

#Followed by Assignmnet2_Processing.py
#featureSets= [RP,SSD,RH,TSSD,TRH,MVD,MT,JS,JSD,JMM,JAM,PLC,MFCC]

JMM = spark.read.orc("hdfs:///user/xzh216/Assign2/output/JMM.orc")
JMM.show(2,False)

# msd-jmir-methods-of-moments-all-v1.0:JMM:11:994623    leaset rows 
JMM.rdd.getNumPartitions()  #8
JMM=JMM.repartition(partitions)
JMM.cache()
JMM.rdd.getNumPartitions() #24



from pyspark.ml.stat import Correlation
from pyspark.ml.feature import VectorAssembler 

###########################################################################################################################
#==============Q1a   Produce descriptive statistics for each feature column in the dataset you picked. Are any  features strongly correlated?
#Descriptive statistics  
JMM_feature= JMM.drop('TRACK_ID')
len(JMM_feature.columns)  #10

statistics = (
    JMM_feature
    .select([col for col in JMM_feature.columns])
    .describe()
    .toPandas()
    .set_index("summary")
    .rename_axis(None)
    .T
)
print(statistics)


#correlations

#VectorAssembler：A feature transformer that merges multiple columns into a vector column
assembler = VectorAssembler(
    inputCols=[col for col in JMM_feature.columns ],
    outputCol="Features"
)
JMM_feature = assembler.transform(JMM_feature)
JMM_feature.cache()
JMM_feature.show(10, True)

pearsonCorr = Correlation.corr(JMM_feature, 'Features', 'pearson').collect()[0][0].toArray()
print(str(pearsonCorr).replace('nan', 'NaN')) 

for i in range(0, pearsonCorr.shape[0]):
    for j in range(i + 1, pearsonCorr.shape[1]):
        if pearsonCorr[i, j] > 0.7:
            print((i, j))

# (0, 5)
# (1, 2)
# (2, 3)
# (3, 4)
# (6, 7)
# (7, 8)
# (8, 9)


###########################################################################################################################
#==============Q1 b   Load the MSD All Music Genre Dataset (MAGD). Visualize the distribution of genres for the songs that were matched.
 
#Load the MSD All Music Genre Dataset (MAGD).

schema_MAGD= StructType([
    StructField('TRACK_ID', StringType()),
    StructField('Genre_Name', StringType())]) 


MAGD = (
    spark.read.format("com.databricks.spark.csv")
    .option("header", "false")
    .option("inferSchema", "false")
    .option("delimiter", "\t")
    .schema(schema_MAGD)
    .load("/data/msd/genre/msd-MAGD-genreAssignment.tsv")
    .repartition(partitions)
)
MAGD.cache()
MAGD.count() # 422714 
#The dataset has a size of 422714 labels.

#Visualize the distribution of genres for the songs that were matched
MAGD.show(2,False)

truemismatched = spark.read.orc("hdfs:///user/xzh216/Assign2/output/truemismatched.orc")
truemismatched.show(2,False)
 
MAGD_Clean = MAGD.join(truemismatched, how='left_anti', on='TRACK_ID')
MAGD_Clean.show(5,25) 
MAGD_Clean.count()  #415350    MAGD.count() # 422714 


import matplotlib.pyplot as plt
(MAGD_Clean 
   .groupBy('Genre_Name') 
   .count()  
   .orderBy('count', ascending=False)
).show(21,False)  
#the same as http://www.ifs.tuwien.ac.at/mir/msd/MAGD.html
# Genre_Name    |count 
genre_distribution = (
     MAGD_Clean 
    .groupBy('Genre_Name') 
    .count()  
    .orderBy('count', ascending=False)  
    .toPandas()
    )
    
x=range(len(genre_distribution["count"]))
y=genre_distribution["count"]
plt.figure(figsize=(15,12)) 
plt.bar(x, y,tick_label=genre_distribution["Genre_Name"])
for a,b in zip(x,y):
    plt.text(a, b+0.05, '%.0f' % b, ha='center', va= 'bottom',fontsize=7)
plt.title('Distribution of Genres for the songs') 
plt.xticks(rotation=45)
plt.xlabel('Genre')
plt.ylabel('Number of Tracks')
plt.savefig('genre_distribution_clean.png')

plt.clf()
plt.figure(figsize=(15,15)) 
plt.pie(y,labels=genre_distribution["Genre_Name"],autopct='%1.2f%%') 
plt.title('Proportion of genres')  
plt.savefig('genre_Proportion_clean.png')




###########################################################################################################################
#==============Q1 c   Merge the genres dataset and the audio features dataset so that every song has a label. 

JMM.count()  # 994623
MAGD_Clean.count()  #415350
 
JMM_labled=JMM.join(MAGD_Clean,on='TRACK_ID', how='inner')    # 
JMM_labled.show(2,False)
JMM_labled.count()  #413293
#413293  TRACKs  has genres label


###########################################################################################################################
#==============Q2   
#  Research and choose three classification algorithms from the spark.ml library.
# Justify your choice of algorithms, taking into account considerations such as explainability,
# interpretability, predictive accuracy, training speed, hyperparameter tuning, dimensionality,and issues with scaling.
# Based on the descriptive statistics from Q1 part (a), decide what processing you shouldn apply to the audio features before using them to train each model.


import numpy as np
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.evaluation import BinaryClassificationEvaluator
from pyspark.ml.feature import VectorAssembler, PCA, StringIndexer, StandardScaler
from pyspark.ml import Pipeline
from pyspark.ml.classification import LogisticRegression, RandomForestClassifier, LinearSVC, GBTClassifier
from pyspark.ml.evaluation import BinaryClassificationEvaluator, MulticlassClassificationEvaluator
from pyspark.ml.tuning import ParamGridBuilder, CrossValidator

# LogisticRegression
# LinearSVC
# RandomForestClassifier

#==============Q2 b  ================================================
#Convert the genre column into a column representing if the song is ”Rap” or some other genre as a binary label.

# creat binary label for Rap genre
JMM_binary= (
    JMM_labled
    .withColumn('Class', F.when(F.col('Genre_Name') == 'Rap', 1).otherwise(0))
    .drop('TRACK_ID')
    )
JMM_binary.cache()
JMM_binary.show(2,False)
#--------|Genre_Name|Class|      #413293


#What is the class balance of the binary label?
JMM_binary.groupBy('Class').count().show()
# +-----+------+
# |Class| count|
# +-----+------+
# |    1| 20566|
# |    0|392727|
# +-----+------+

 
def print_class_balance(data, name):
    N = data.count()
    counts = data.groupBy("Class").count().toPandas()
    counts["ratio"] = counts["count"] / N
    print(name)
    print(N)
    print(counts)
    print("")

print_class_balance(JMM_binary,"all features")
# all features
# 413293
   # Class   count     ratio
#       1   20566  0.049761
#       0  392727  0.950239


###########################################################################################################################
def print_binary_metrics(predictions, labelCol="Class", predictionCol="prediction", rawPredictionCol="rawPrediction"):
    total = predictions.count()
    positive = predictions.filter((F.col(labelCol) == 1)).count()
    negative = predictions.filter((F.col(labelCol) == 0)).count()
    nP = predictions.filter((F.col(predictionCol) == 1)).count()
    nN = predictions.filter((F.col(predictionCol) == 0)).count()
    TP = predictions.filter((F.col(predictionCol) == 1) & (F.col(labelCol) == 1)).count()
    FP = predictions.filter((F.col(predictionCol) == 1) & (F.col(labelCol) == 0)).count()
    FN = predictions.filter((F.col(predictionCol) == 0) & (F.col(labelCol) == 1)).count()
    TN = predictions.filter((F.col(predictionCol) == 0) & (F.col(labelCol) == 0)).count()

    binary_evaluator = BinaryClassificationEvaluator(rawPredictionCol="rawPrediction", labelCol=labelCol, metricName="areaUnderROC")
    auroc = binary_evaluator.evaluate(predictions)

    print('actual total:    {}'.format(total))
    print('actual positive: {}'.format(positive))
    print('actual negative: {}'.format(negative))
    print('nP:              {}'.format(nP))
    print('nN:              {}'.format(nN))
    print('TP:              {}'.format(TP))
    print('FP:              {}'.format(FP))
    print('FN:              {}'.format(FN))
    print('TN:              {}'.format(TN))
    print('precision:       {}'.format(TP / (TP + FP)))
    print('recall:          {}'.format(TP / (TP + FN)))
    print('accuracy:        {}'.format((TP + TN) / total))
    print('auroc:           {}'.format(auroc))

def with_custom_prediction(predictions, threshold):

    def apply_custom_threshold(probability, threshold):
        return int(probability[1] > threshold)

    apply_custom_threshold_udf = F.udf(lambda x: apply_custom_threshold(x, threshold), IntegerType())

    return predictions.withColumn("customPrediction", apply_custom_threshold_udf(F.col("probability")))


def Generate_Train_Test(pipeline,trainRaw,testRaw): 
    # fit pipeline on training data
    pipeline_model = pipeline.fit(trainRaw)
    # transform train and test data   #--------|Genre_Name|Class|
    train = pipeline_model.transform(trainRaw).select('Genre_Name', 'Features', 'Class')
    test = pipeline_model.transform(testRaw).select('Genre_Name', 'Features', 'Class')
    return train,test
 
 
#==============Q2 a  c  ================================================
# Split the dataset into training and test sets. Note that you may need to take class balance
# into account using a sampling method such as stratification, subsampling, or oversampling.
# Justify your choice of sampling method.

 
assembler = VectorAssembler(
    inputCols=[col for col in JMM_binary.columns if col.startswith("JMM_")],
    outputCol="raw_Features"
)
JMM_binary_Vfeature = assembler.transform(JMM_binary) 
JMM_binary_Vfeature.cache() 
JMM_binary_Vfeature.show() 
#|Genre_Name|Class|        raw_Features|

print_class_balance(JMM_binary_Vfeature, "alldata")
# alldata
# 413293
   # Class   count     ratio
# 0      1   20566  0.049761
# 1      0  392727  0.950239



# 20566	  vs   392727
 
#way 1===randomSplit (not stratified)
training_random, test_random = JMM_binary_Vfeature.randomSplit([0.8, 0.2], seed = 1000)
training_random.cache()
test_random.cache()

print_class_balance(training_random, "training")
print_class_balance(test_random, "test")

# training
# 331110
   # Class   count     ratio
# 0      1   16520  0.049893
# 1      0  314590  0.950107

# test
# 82183
   # Class  count     ratio
# 0      1   4046  0.049232
# 1      0  78137  0.950768

from pyspark.sql.window import *
from pyspark.sql.functions import monotonically_increasing_id  
# way 2 -1  stratified sample  =proportional sampling
# sampleBy(col, fractions, seed=None)¶
# Returns a stratified sample without replacement based on the fraction given on each stratum
# fractions – sampling fraction for each stratum. If a stratum is not specified, we treat its fraction as zero.
 
 
temp = JMM_binary_Vfeature.withColumn("id", monotonically_increasing_id())
training_stratified = temp.sampleBy("Class", fractions={0: 0.8, 1: 0.8}, seed = 1000)
training_stratified.cache()

test_stratified = temp.join(training_stratified, on="id", how="left_anti")
test_stratified.cache()

training_stratified = training_stratified.drop("id")
test_stratified = test_stratified.drop("id")
 
print_class_balance(training_stratified, "training")
print_class_balance(test_stratified, "test")
# training
# 331110
   # Class   count     ratio
# 0      1   16382  0.049476
# 1      0  314728  0.950524

# test
# 82183
   # Class  count     ratio
# 0      1   4184  0.050911
# 1      0  77999  0.949089
 
 
#way 2-2 Exact stratification using Window     ===multi-class variant in comments

temp = (
    JMM_binary_Vfeature
    .withColumn("id", F.monotonically_increasing_id())
    .withColumn("Random", F.rand(seed=1000))
    .withColumn(
        "Row",
        F.row_number()
        .over(
            Window
            .partitionBy("Class")
            .orderBy("Random")
        )
    )
)
#top 20899 rows are class  1
 
num_P=20566	   
num_N=392727
training_stratification = temp.where(
    ((F.col("Class") == 0) & (F.col("Row") < num_N * 0.8)) |      
    ((F.col("Class") == 1) & (F.col("Row") < num_P	* 0.8))
)
training_stratification.cache()

test_stratification = temp.join(training_stratification, on="id", how="left_anti")
test_stratification.cache()
print_class_balance(training_stratification, "training_stratification")
print_class_balance(test_stratification, "test_stratification")


# training_stratification
# 330633
   # Class   count     ratio
# 0      1   16452  0.049759
# 1      0  314181  0.950241

# test_stratification
# 82660
   # Class  count    ratio
# 0      1   4114  0.04977
# 1      0  78546  0.95023




# way3  Downsampling    50:50
training_downsampled =(
      training_random
     .withColumn("Random", F.rand(seed=1000))
     .where((F.col("Class") != 0) | ((F.col("Class") == 0) & (F.col("Random") < 2 * (num_P / num_N))))
     )
   
training_downsampled.cache()
test_downsampled=test_random

print_class_balance(training_downsampled, "training_downsampled")
print_class_balance(test_downsampled, "test_downsampled")

# training_downsampled===1
# 32859
   # Class  count     ratio
# 0      1  16520  0.502754
# 1      0  16339  0.497246

# test_downsampled
# 82183
   # Class  count     ratio
# 0      1   4046  0.049232
# 1      0  78137  0.950768




# training_downsampled==2
# 49337
   # Class  count    ratio
# 0      1  16520  0.33484
# 1      0  32817  0.66516

# test_downsampled
# 82183
   # Class  count     ratio
# 0      1   4046  0.049232
# 1      0  78137  0.950768







 
print(np.random.random(20))
#way4  Upsampling     wrong 
#Randomly upsample by exploding a vector of length betwen 0 and n for each row

ratio = 10
n = 20
p = ratio / n  # ratio < n such that probability < 1

#np.random.random(n)=> generate N numbers(0,10)  =>  p (0.5) 
def random_resample(x, n, p):
    # Can implement custom sampling logic per class,
    if x == 0:
        return [0]  # no sampling
    if x == 1:
        return list(range((np.sum(np.random.random(n) > p))))  # upsampling      the smaller p  the more positvie observations   p=1  all  negative
    return []  # drop
#explode(col)  Returns a new row for each element in the given array or map.
random_resample_udf = F.udf(lambda x: random_resample(x, n, p), ArrayType(IntegerType()))
training_upsampled = (
    training_random
    .withColumn("Sample", random_resample_udf(F.col("Class")))
     .select(
        F.col("Genre_Name"),
        F.col("Class"),
        F.col("raw_Features"),
        F.explode(F.col("Sample")).alias("Sample")
        )
       .drop("Sample")
)
 
test_upsampled=test_random
print_class_balance(training_upsampled, "training_upsampled")
print_class_balance(test_upsampled, "test_upsampled")
# 479135
   # Class   count     ratio
# 0      1  164970  0.344308
# 1      0  314590  0.656579

 
 
# way5 Observation reweighting 
training_weighted = (
    training_random
    .withColumn(
        "Weight",F.when(F.col("Class") == 0, 1.0)
          .otherwise(10.0)   #Class==1   10 
    )
)
test_weighted=test_random
 

#=======Justify the choice of sampling method  focusing on  logistic regression
 
 
#logistic regression
#choose the proper K for PCA   #using for training data only   eg training_random
standard_scaler = StandardScaler(inputCol="raw_Features", outputCol="scaled_features")
standard_fit = standard_scaler.fit(training_random)
standard_train=standard_fit.transform(training_random)
pca = PCA(k=10, inputCol="scaled_features", outputCol="pca_features") 
model_pca = pca.fit(standard_train)
#model_pca = pca.fit(standard_train) 
#model_pca.explainedVariance   #Returns a vector of proportions of variance explained by each principal component.
tt=[round(num,3) for num in model_pca.explainedVariance]
print(tt) 
#[0.422, 0.306, 0.149, 0.071, 0.024, 0.015, 0.007, 0.004, 0.002, 0.0]
plt.figure()
plt.plot(range(1,11),model_pca.explainedVariance)
plt.xlabel('Number of Components')
plt.ylabel('Variance (%)') 
plt.title('Explained Variance')
plt.savefig('PCA.png')
 

 
 
#choose the first 5  components 
standard_scaler = StandardScaler(inputCol="raw_Features", outputCol="scaled_features")
pca = PCA(k=5, inputCol="scaled_features", outputCol="Features")  #make sure the last name is "Features"
pipeline_Logic = Pipeline(stages=[standard_scaler,pca])


#TAKE LogisticRegression,find the best samping methods
#fit pipeline(including pca)  on  training data only    and   project /transform  on  testing  data

train_data,test_data = Generate_Train_Test(pipeline_Logic,training_random,test_random) 
 
lr = LogisticRegression(featuresCol='Features', labelCol='Class')
lr_model = lr.fit(train_data)   #train
predictions = lr_model.transform(test_data)  #test
predictions.cache()
predictions.show(5,False)

predictions = with_custom_prediction(predictions, 0.3)
print_binary_metrics(predictions, predictionCol="customPrediction")
 
 
 
 
 
 
train_data,test_data = Generate_Train_Test(pipeline_Logic,training_stratification,test_stratification) 
lr = LogisticRegression(featuresCol='Features', labelCol='Class')
lr_model = lr.fit(train_data)   #train
predictions = lr_model.transform(test_data)  #test
predictions.cache()
predictions = with_custom_prediction(predictions, 0.5)  #0.3
print_binary_metrics(predictions, predictionCol="customPrediction")

 




 
train_data,test_data = Generate_Train_Test(pipeline_Logic,training_upsampled,test_upsampled) 
lr = LogisticRegression(featuresCol='Features', labelCol='Class')
lr_model = lr.fit(train_data)   #train
predictions = lr_model.transform(test_data)  #test
predictions.cache()
predictions = with_custom_prediction(predictions, 0.5)   
print_binary_metrics(predictions, predictionCol="customPrediction")




 
 
 
pipeline_model = pipeline_Logic.fit(training_weighted)
train_data = pipeline_model.transform(training_weighted).select('Genre_Name', 'Features', 'Class',"Weight")
test_data = pipeline_model.transform(test_weighted).select('Genre_Name', 'Features', 'Class')
lr = LogisticRegression(featuresCol='Features', labelCol='Class', weightCol="Weight")
lr_model = lr.fit(train_data)
predictions = lr_model.transform(test_data)
predictions.cache()
predictions = with_custom_prediction(predictions, 0.5)   
print_binary_metrics(predictions, predictionCol="customPrediction")
 
 
 
 
 
training_random.count()
training_downsampled.count()
 
 
from datetime import datetime 
a=datetime.now()  
train_data,test_data = Generate_Train_Test(pipeline_Logic,training_downsampled,test_downsampled)    #LogisticRegression==PCA
lr = LogisticRegression(featuresCol='Features', labelCol='Class')
lr_model = lr.fit(train_data)   #train
predictions = lr_model.transform(test_data)  #test
predictions.cache()
predictions = with_custom_prediction(predictions, 0.5)   
print_binary_metrics(predictions, predictionCol="customPrediction")
predictions = with_custom_prediction(predictions, 0.6)   
print_binary_metrics(predictions, predictionCol="customPrediction")
b=datetime.now() 
print((b-a).seconds) #5
 
 

 
##================down sampling   has the highest recall and auroc 
  
  
#==============Q2  d  e ================================================

 
#============LogisticRegression
# class pyspark.ml.classification.LogisticRegression(featuresCol='features', labelCol='label', predictionCol='prediction',
# maxIter=100, regParam=0.0, elasticNetParam=0.0, tol=1e-06, fitIntercept=True, threshold=0.5, thresholds=None, 
# probabilityCol='probability', rawPredictionCol='rawPrediction', standardization=True, weightCol=None, aggregationDepth=2, family='auto', 
# lowerBoundsOnCoefficients=None, upperBoundsOnCoefficients=None, lowerBoundsOnIntercepts=None, upperBoundsOnIntercepts=None)

# For alpha = 0, the penalty is an L2 penalty. For alpha = 1, it is an L1 penalty.'

from datetime import datetime 
a=datetime.now()  
lr = LogisticRegression(featuresCol='raw_Features', labelCol='Class')    #elasticNetParam 0  L2 
lr_model = lr.fit(training_downsampled)   #train
predictions = lr_model.transform(test_downsampled)  #test
predictions.cache()
print_binary_metrics(predictions)
b=datetime.now() 
print((b-a).seconds)  


# actual total:    82183
# actual positive: 4046
# actual negative: 78137
# nP:              11436
# nN:              70747
# TP:              2555
# FP:              8881
# FN:              1491
# TN:              69256
# precision:       0.22341727876880027
# recall:          0.6314878892733564
# accuracy:        0.8737938503096747
# auroc:           0.858413849659377
# 19
 
#============Linear SVM Classifier
# LinearSVC(featuresCol='features', labelCol='label', predictionCol='prediction',
# maxIter=100, regParam=0.0, tol=1e-06, rawPredictionCol='rawPrediction', fitIntercept=True, standardization=True,
# threshold=0.0, weightCol=None, aggregationDepth=2) 
#This binary classifier optimizes the Hinge Loss using the OWLQN optimizer. Only supports L2 regularization currently.
 
  
#standardization=True
a=datetime.now() 
svm = LinearSVC(featuresCol='raw_Features', labelCol='Class')   
svm_model = svm.fit(training_downsampled)     
predictions = svm_model.transform(test_downsampled)  #test
predictions.cache()
print_binary_metrics(predictions)
b=datetime.now() 
print((b-a).seconds)     
 
 

# actual total:    82183
# actual positive: 4046
# actual negative: 78137
# nP:              12611
# nN:              69572
# TP:              2653
# FP:              9958
# FN:              1393
# TN:              68179
# precision:       0.21037189754975816
# recall:          0.6557093425605537
# accuracy:        0.8618814109974082
# auroc:           0.849991054028584
# 36

 
  
#=================  LinearSVC  is  wores  than LogisticRegression  


#=============    RandomForestClassifier
#class pyspark.ml.classification.RandomForestClassifier(featuresCol='features', labelCol='label',
# predictionCol='prediction', probabilityCol='probability', rawPredictionCol='rawPrediction', maxDepth=5, maxBins=32, 
#     minInstancesPerNode=1, minInfoGain=0.0, maxMemoryInMB=256, cacheNodeIds=False, checkpointInterval=10, impurity='gini', 
#     numTrees=20, featureSubsetStrategy='auto', seed=None, subsamplingRate=1.0)
# Random Forest learning algorithm for classification. It supports both binary and multiclass labels, as well as both continuous and categorical features.
a=datetime.now() 
rf = RandomForestClassifier(featuresCol='raw_Features', labelCol='Class')   
rf_model = rf.fit(training_downsampled)     
predictions = rf_model.transform(test_downsampled)   
predictions.cache()
print_binary_metrics(predictions)
b=datetime.now() 
print((b-a).seconds)   #5   
  


# actual total:    82183
# actual positive: 4046
# actual negative: 78137
# nP:              12925
# nN:              69258
# TP:              2677
# FP:              10248
# FN:              1369
# TN:              67889
# precision:       0.20711798839458415
# recall:          0.6616411270390509
# accuracy:        0.8586447318788557
# auroc:           0.8536950411020927
# 5

 
 
#============GBTClassifier
# class pyspark.ml.classification.GBTClassifier(featuresCol='features', labelCol='label', predictionCol='prediction',
# maxDepth=5, maxBins=32, minInstancesPerNode=1, minInfoGain=0.0, maxMemoryInMB=256, cacheNodeIds=False, checkpointInterval=10, 
# lossType='logistic', maxIter=20, stepSize=0.1, seed=None, subsamplingRate=1.0, featureSubsetStrategy='all') 
# Gradient-Boosted Trees (GBTs) learning algorithm for classification. It supports binary labels, as well as both continuous and categorical features.


# maxBins = Param(parent='undefined', name='maxBins', doc='Max number of bins for discretizing continuous features. Must be >=2 and >= number of categories for any categorical feature.')
# maxDepth = Param(parent='undefined', name='maxDepth', doc='Maximum depth of the tree. (>= 0) E.g., depth 0 means 1 leaf node; depth 1 means 1 internal node + 2 leaf nodes.')¶
# minInstancesPerNode = Param(parent='undefined', name='minInstancesPerNode', 
# doc='Minimum number of instances each child must have after split. If a split causes the left or right child to have fewer than minInstancesPerNode, 
# the split will be discarded as invalid. Should be >= 1.')¶

a=datetime.now() 
gbt = GBTClassifier(featuresCol='raw_Features', labelCol='Class')   
gbt_model = gbt.fit(training_downsampled)     
predictions = gbt_model.transform(test_downsampled)   
predictions.cache()
print_binary_metrics(predictions)
b=datetime.now() 
print((b-a).seconds)    

 
# actual total:    82183
# actual positive: 4046
# actual negative: 78137
# nP:              13072
# nN:              69111
# TP:              2772
# FP:              10300
# FN:              1274
# TN:              67837
# precision:       0.2120563035495716
# recall:          0.6851211072664359
# accuracy:        0.859167954443133
# auroc:           0.8689375884281348
# 25


#============GBTClassifier has the best  performance   but  slow then RF  AND  low  explainability
 
 
###########################################################################################################################
#==============  Q3 ============ 
# (a) Look up the hyperparameters for each of your classification algorithms. Try to understand
# how these hyperparameters affect the performance of each model and if the values you
# used in Q2 part (d) were sensible.
# (b) Use cross-validation to tune some of the hyperparameters of your best performing binary
# classification model.
# How has this changed your performance metrics?
 
model_list = ['LogisticRegression', 'LinearSVC', 'RandomForest','GradientBoostedTrees'] 

# rename class to  label    becasue   IllegalArgumentException: 'Field "label" does not exist.\nAvailable fields:
# estimator in cv.fit  only recognize hte  default label name "label"
training_downsampled=training_downsampled.withColumnRenamed("Class", "label")
test_downsampled=test_downsampled.withColumnRenamed("Class", "label") 
 
 
# maxIter=100, regParam=0.0, elasticNetParam=0.0, tol=1e-06, fitIntercept=True, threshold=0.5,  aggregationDepth=2,
lr = LogisticRegression(featuresCol='raw_Features', labelCol='label')  

# maxIter=100, regParam=0.0, tol=1e-06,  threshold=0.0,  aggregationDepth=2) 
svm = LinearSVC(featuresCol='raw_Features', labelCol='label') #default standardization=True ,regParam=0

# maxDepth=5, maxBins=32, minInstancesPerNode=1, minInfoGain=0.0, maxMemoryInMB=256, 
# cacheNodeIds=False, checkpointInterval=10, numTrees=20, 

rf = RandomForestClassifier(featuresCol='raw_Features', labelCol='label')

gbt = GBTClassifier(featuresCol='raw_Features', labelCol='label') 
 
estimators_list = [lr,svm, rf,gbt]
    
#create list of parameter grid
lr_paramGrid = (ParamGridBuilder()
             .addGrid(lr.regParam, [0,0.01,0.1]) # regularization parameter
             .addGrid(lr.elasticNetParam, [0,1]) # Elastic Net Parameter
             .build())

svm_paramGrid = (ParamGridBuilder()
             .addGrid(svm.regParam, [0,0.01,0.1]) # regularization parameter
             .build())
      
             
rf_paramGrid = (ParamGridBuilder()
             .addGrid(rf.numTrees, [20,50,100]) 
             .addGrid(rf.maxDepth, [2,4,8]) 
             .addGrid(rf.maxBins, [16,32,64])
             .build())                
 
gbt_paramGrid = (ParamGridBuilder()
             .addGrid(gbt.maxDepth, [2, 5, 8])   #5 is defaultmaxDepth
             .addGrid(gbt.maxBins, [16, 32, 64])
             .addGrid(gbt.stepSize, [0.01, 0.1])
             .build())
 
paramGrids = [lr_paramGrid,svm_paramGrid,rf_paramGrid,gbt_paramGrid]

 
binary_evaluator = BinaryClassificationEvaluator(metricName="areaUnderROC") #default areaUnderROC
muti_evaluator = MulticlassClassificationEvaluator() # default 'f1'   (f1|weightedPrecision|weightedRecall|accuracy)') 
 
# run cv and check the performance 


#evaluator=areaUnderROC
finalModel_list_b = []    
for i in range(len(estimators_list)):
    cv = CrossValidator(
                    estimator=estimators_list[i],
                    estimatorParamMaps=paramGrids[i],
                    evaluator=binary_evaluator,  #areaUnderROC 
                    numFolds=10,seed=1000, parallelism=8)
    cv_model = cv.fit(training_downsampled)   #train 
    cv_prediction = cv_model.transform(test_downsampled) #test
    finalModel_list_b.append(cv_model.bestModel) #save the best model
    
    print("------Cross-Validation performance of {}:--------".format(model_list[i]))
    print_binary_metrics(cv_prediction,labelCol="label")   ##'Field "Class" does not exist.\n
    print("--------------------------------------------------------------")
  
# ------Cross-Validation performance of LogisticRegression:--------
# actual total:    82183
# actual positive: 4046
# actual negative: 78137
# nP:              11436
# nN:              70747
# TP:              2555
# FP:              8881
# FN:              1491
# TN:              69256
# precision:       0.22341727876880027
# recall:          0.6314878892733564
# accuracy:        0.8737938503096747
# auroc:           0.8584138464962445
# --------------------------------------------------------------
# ------Cross-Validation performance of LinearSVC:--------
# actual total:    82183
# actual positive: 4046
# actual negative: 78137
# nP:              12085
# nN:              70098
# TP:              2600
# FP:              9485
# FN:              1446
# TN:              68652
# precision:       0.21514273893256103
# recall:          0.6426099851705388
# accuracy:        0.86699195697407
# auroc:           0.8520686848797557
# --------------------------------------------------------------
# ------Cross-Validation performance of RandomForest:--------
# actual total:    82183
# actual positive: 4046
# actual negative: 78137
# nP:              12792
# nN:              69391
# TP:              2770
# FP:              10022
# FN:              1276
# TN:              68115
# precision:       0.216541588492808
# recall:          0.6846267918932278
# accuracy:        0.8625263132277965
# auroc:           0.869017296204794
# --------------------------------------------------------------

# ------Cross-Validation performance of GradientBoostedTrees:--------
# actual total:    82183
# actual positive: 4046
# actual negative: 78137
# nP:              12958
# nN:              69225
# TP:              2793
# FP:              10165
# FN:              1253
# TN:              67972
# precision:       0.2155425219941349
# recall:          0.6903114186851211
# accuracy:        0.8610661572344646
# auroc:           0.8717152948421343
# --------------------------------------------------------------
 
 
  
###########################################################################################################################
#==============  Q4  ============ 

#Convert the genre column into an integer index that encodes each genre consistently.
# JMM_labled :df after joining  MAGD_Clean      

label_stringIdx = StringIndexer(inputCol = "Genre_Name", outputCol = "label")
JMM_multi = label_stringIdx.fit(JMM_labled).transform(JMM_labled)
JMM_multi = JMM_multi.drop('TRACK_ID')
 
# Convert the genre column into an integer index that encodes each genre consistently. Find
# a way to do this that requires the least amount of work by hand
 
assembler = VectorAssembler(
    inputCols=[col for col in JMM_multi.columns if col.startswith("JMM_")],
    outputCol="raw_Features"
)
JMM_multi_Vfeature = assembler.transform(JMM_multi) 
JMM_multi_Vfeature.cache() 
JMM_multi_Vfeature.show() 
 
 
 
 
# 
def print_class_balance(data, name):
    N = data.count()
    counts = data.groupBy("label").count().orderBy("count").toPandas()
    counts["ratio"] = counts["count"] / N
    print(name)
    print(N)
    print(counts)
    print("")
 
    
training_random, test_random = JMM_multi_Vfeature.randomSplit([0.8, 0.2], seed = 1000)     
print_class_balance(training_random,"training_random")
#label=0  has 57% 

#=============1 train a sigal  
lr = LogisticRegression(featuresCol='raw_Features', labelCol='label')    #elasticNetParam 0  L2 
lr_model = lr.fit(training_random)   #train
predictions = lr_model.transform(test_random)  #test
predictions.cache()

#train result
trainingSummary = lr_model.summary
accuracy = trainingSummary.accuracy
falsePositiveRate = trainingSummary.weightedFalsePositiveRate
truePositiveRate = trainingSummary.weightedTruePositiveRate
fMeasure = trainingSummary.weightedFMeasure()
precision = trainingSummary.weightedPrecision
recall = trainingSummary.weightedRecall
print("Accuracy: %s\nFPR: %s\nTPR: %s\nF-measure: %s\nPrecision: %s\nRecall: %s"
       % (accuracy, falsePositiveRate, truePositiveRate, fMeasure, precision, recall))

print("F1: ", muti_evaluator.evaluate(predictions, {muti_evaluator.metricName: "f1"}))
print('Accuracy:', muti_evaluator.evaluate(predictions, {muti_evaluator.metricName: "accuracy"}))
print("WeightedPrecision: ", muti_evaluator.evaluate(predictions, {muti_evaluator.metricName: "weightedPrecision"}))
print('WeightedRecall:', muti_evaluator.evaluate(predictions, {muti_evaluator.metricName: "weightedRecall"}))
 
# F1:  0.45239870381199754
# Accuracy: 0.5740724967450689
# WeightedPrecision:  0.4097525386191115
# WeightedRecall: 0.5740724967450689

rf = RandomForestClassifier(featuresCol='raw_Features', labelCol='label')   
rf_model = rf.fit(training_random)     
predictions = rf_model.transform(test_random)  #test
predictions.cache()
print("F1: ", muti_evaluator.evaluate(predictions, {muti_evaluator.metricName: "f1"}))
print('Accuracy:', muti_evaluator.evaluate(predictions, {muti_evaluator.metricName: "accuracy"}))
print("WeightedPrecision: ", muti_evaluator.evaluate(predictions, {muti_evaluator.metricName: "weightedPrecision"}))
print('WeightedRecall:', muti_evaluator.evaluate(predictions, {muti_evaluator.metricName: "weightedRecall"}))

 
# F1:  0.4273786581231458
# Accuracy: 0.5716023995230157
# WeightedPrecision:  0.373629058955659
# WeightedRecall: 0.5716023995230157


#===========2 Cross Validation   Logistic classifier
cv = CrossValidator(
                    estimator=lr,
                    estimatorParamMaps=lr_paramGrid,
                    evaluator=muti_evaluator,  # 
                    numFolds=10,seed=1000, parallelism=8 )
cv_model = cv.fit(training_random)   #train 
#focusing on test 
cv_prediction = cv_model.transform(test_random) #test
        
print("-----performance of LogisticRegression --------")
print("F1: ", muti_evaluator.evaluate(cv_prediction, {muti_evaluator.metricName: "f1"}))
print('Accuracy:', muti_evaluator.evaluate(cv_prediction, {muti_evaluator.metricName: "accuracy"}))
print("WeightedPrecision: ", muti_evaluator.evaluate(cv_prediction, {muti_evaluator.metricName: "weightedPrecision"}))
print('WeightedRecall:', muti_evaluator.evaluate(cv_prediction, {muti_evaluator.metricName: "weightedRecall"}))
print("--------------------------------------------------------------")
(#f1|weightedPrecision|weightedRecall|accuracy)')
 
# ------Cross-Validation performance of LogisticRegression --------
# F1:  0.45239870381199754
# Accuracy: 0.5740724967450689
# WeightedPrecision:  0.4097525386191115
# WeightedRecall: 0.5740724967450689
# --------------------------------------------------------------

#===========3 Cross Validation Random Forest

cv = CrossValidator(
                    estimator=rf,
                    estimatorParamMaps=rf_paramGrid,
                    evaluator=muti_evaluator,  # 
                    numFolds=10,seed=1000, parallelism=8 )
cv_model = cv.fit(training_random)   #train 
cv_prediction = cv_model.transform(test_random) #test
        
print("------Cross-Validation performance of RandomForest-------")
print("F1: ", muti_evaluator.evaluate(cv_prediction, {muti_evaluator.metricName: "f1"}))
print('Accuracy:', muti_evaluator.evaluate(cv_prediction, {muti_evaluator.metricName: "accuracy"}))
print("WeightedPrecision: ", muti_evaluator.evaluate(cv_prediction, {muti_evaluator.metricName: "weightedPrecision"}))
print('WeightedRecall:', muti_evaluator.evaluate(cv_prediction, {muti_evaluator.metricName: "weightedRecall"}))
print("--------------------------------------------------------------")
 
# ------Cross-Validation performance of RandomForest:--------              
# F1:  0.4536149624105848
# Accuracy: 0.5803998393828408
# WeightedPrecision:  0.40184168188238356
# WeightedRecall: 0.5803998393828407

# ----------------------------------------------------------
 

#===========4 try  down sampling
#down sampling  label=0
num_P=186485   
num_N=331110-186485
training_downsampled =(
      training_random.withColumn("Random", F.rand(seed=1000))
     .where((F.col("label") != 0) | ((F.col("label") == 0) & (F.col("Random") >10 * (num_P / num_N))))
     )
print_class_balance(training_downsampled,"training_downsampled")
test_downsampled=test_random
# training_downsampled
# 144625
    # label  count     ratio
# 0    20.0    169  0.001169
# 1    19.0    380  0.002627
# 2    18.0    435  0.003008
#...
# 15    5.0  11267  0.077905
# 16    4.0  13930  0.096318
# 17    3.0  14075  0.097321
# 18    2.0  16520  0.114226
# 19    1.0  32130  0.222161

 
cv = CrossValidator(
                    estimator=rf,
                    estimatorParamMaps=rf_paramGrid,
                    evaluator=muti_evaluator,  # 
                    numFolds=10,seed=1000, parallelism=8 )
cv_model = cv.fit(training_downsampled)   #train 
cv_prediction = cv_model.transform(test_downsampled) #test
        
print("------Cross-Validation performance of RandomForest -------")
print("F1: ", muti_evaluator.evaluate(cv_prediction, {muti_evaluator.metricName: "f1"}))
print('Accuracy:', muti_evaluator.evaluate(cv_prediction, {muti_evaluator.metricName: "accuracy"}))
print("WeightedPrecision: ", muti_evaluator.evaluate(cv_prediction, {muti_evaluator.metricName: "weightedPrecision"}))
print('WeightedRecall:', muti_evaluator.evaluate(cv_prediction, {muti_evaluator.metricName: "weightedRecall"}))
print("--------------------------------------------------------------")


JMM_multi.select("Genre_Name").distinct().count()#21
JMM_multi.select("label").distinct().count()

label_stringIdx = StringIndexer(inputCol = "Genre_Name", outputCol = "label")
JMM_multi = label_stringIdx.fit(JMM_labled).transform(JMM_labled)

#=========5 try mllib    ==============================
from pyspark.mllib.classification import LogisticRegressionWithLBFGS, LogisticRegressionModel,LogisticRegressionWithSGD
from pyspark.mllib.regression import LabeledPoint
from pyspark.mllib.tree import RandomForest, RandomForestModel
from pyspark.mllib.util import MLUtils
from pyspark.mllib.evaluation import MulticlassMetrics   #RDD-based API
 

JMM_multi_Vfeature.show()
training_random, test_random = JMM_multi_Vfeature.randomSplit([0.8, 0.2], seed = 1000)  
training_random.show()
print_class_balance(training_random,"training_random")
# training_random
# 331110
    # label   count     ratio
# 0    20.0     169  0.000510
# 1    19.0     380  0.001148
# 2    18.0     435  0.001314
# 3    17.0     818  0.002470
# 4    16.0    1243  0.003754
# 5    15.0    1305  0.003941
# 6    14.0    1637  0.004944
# 7    13.0    3144  0.009495
# 8    12.0    4515  0.013636
# 9    11.0    4852  0.014654
# 10   10.0    5433  0.016408
# 11    9.0    5535  0.016716
# 12    8.0    6948  0.020984
# 13    7.0    9189  0.027752
# 14    6.0   11100  0.033524
# 15    5.0   11267  0.034028
# 16    4.0   13930  0.042071
# 17    3.0   14075  0.042509
# 18    2.0   16520  0.049893
# 19    1.0   32130  0.097037
# 20    0.0  186485  0.563212

#  Genre_Name|label|        raw_Features|
#Create labeledPoints from a Spark DataFrame using Pyspark
training  = training_random.rdd.map(lambda row: LabeledPoint(row['label'], row['raw_Features'].toArray())) 
test  = test_random.rdd.map(lambda row: LabeledPoint(row['label'], row['raw_Features'].toArray()))    
#label  features 

#========LogisticRegressionModel
 
# Run training algorithm to build the model
#lr_model = LogisticRegressionWithSGD.train(sc.parallelize(training), validateData=False)
lr_model = LogisticRegressionWithSGD.train(training, validateData=False)
# Compute raw scores on the test set
predictionAndLabels = test.map(lambda lp: (float(lr_model.predict(lp.features)), lp.label))

# Instantiate metrics object
metrics = MulticlassMetrics(predictionAndLabels)
metrics.confusionMatrix().toArray()

#Overall statistics
print("Recall = %s" % metrics.recall())
print("Precision = %s" % metrics.precision())
print("F1 measure = %s" % metrics.fMeasure())
print("Accuracy = %s" % metrics.accuracy)

# Recall = 0.09641896742635338
# Precision = 0.09641896742635338
# F1 measure = 0.09641896742635338
# Accuracy = 0.09641896742635338

# # Weighted stats
print("Weighted recall = %s" % metrics.weightedRecall)
print("Weighted precision = %s" % metrics.weightedPrecision)
print("Weighted F1 Score = %s" % metrics.weightedFMeasure())
 
# Weighted recall = 0.09641896742635338
# Weighted precision = 0.32910176797518254
# Weighted F1 Score = 0.01778281690109595
 
 
# # Statistics by class
labels = test.map(lambda lp: lp.label).distinct().collect()
for label in sorted(labels):
    print("Class %s precision = %s" % (label, metrics.precision(label)))
    print("Class %s recall = %s" % (label, metrics.recall(label)))
    print("Class %s F1 Measure = %s" % (label, metrics.fMeasure(label, beta=1.0)))


# Class 0.0 precision = 0.5652173913043478
# Class 0.0 recall = 0.0008385293485271985
# Class 0.0 F1 Measure = 0.001674574379012001
# Class 1.0 precision = 0.09602503836130258
# Class 1.0 recall = 0.9984804356084589
# Class 1.0 F1 Measure = 0.17520080879003677
# Class 2.0 precision = 0.0
# Class 2.0 recall = 0.0
# Class 2.0 F1 Measure = 0.0
# Class 3.0 precision = 0.0
# Class 3.0 recall = 0.0
# Class 3.0 F1 Measure = 0.0
# Class 4.0 precision = 0.0
# Class 4.0 recall = 0.0
# Class 4.0 F1 Measure = 0.0
# Class 5.0 precision = 0.0
# Class 5.0 recall = 0.0
# Class 5.0 F1 Measure = 0.0
# Class 6.0 precision = 0.0
# Class 6.0 recall = 0.0
# Class 6.0 F1 Measure = 0.0
# Class 7.0 precision = 0.0
# Class 7.0 recall = 0.0
# Class 7.0 F1 Measure = 0.0
# Class 8.0 precision = 0.0
# Class 8.0 recall = 0.0
# Class 8.0 F1 Measure = 0.0
# Class 9.0 precision = 0.0
# Class 9.0 recall = 0.0
# Class 9.0 F1 Measure = 0.0
# Class 10.0 precision = 0.0
# Class 10.0 recall = 0.0
# Class 10.0 F1 Measure = 0.0
# Class 11.0 precision = 0.0
# Class 11.0 recall = 0.0
# Class 11.0 F1 Measure = 0.0
# Class 12.0 precision = 0.0
# Class 12.0 recall = 0.0
# Class 12.0 F1 Measure = 0.0
# Class 13.0 precision = 0.0
# Class 13.0 recall = 0.0
# Class 13.0 F1 Measure = 0.0
# Class 14.0 precision = 0.0
# Class 14.0 recall = 0.0
# Class 14.0 F1 Measure = 0.0
# Class 15.0 precision = 0.0
# Class 15.0 recall = 0.0
# Class 15.0 F1 Measure = 0.0
# Class 16.0 precision = 0.0
# Class 16.0 recall = 0.0
# Class 16.0 F1 Measure = 0.0
# Class 17.0 precision = 0.0
# Class 17.0 recall = 0.0
# Class 17.0 F1 Measure = 0.0
# Class 18.0 precision = 0.0
# Class 18.0 recall = 0.0
# Class 18.0 F1 Measure = 0.0
# Class 19.0 precision = 0.0
# Class 19.0 recall = 0.0
# Class 19.0 F1 Measure = 0.0
# Class 20.0 precision = 0.0
# Class 20.0 recall = 0.0
# Class 20.0 F1 Measure = 0.0
 
  
training  = training_random.rdd.map(lambda row: LabeledPoint(row['label'], row['raw_Features'].toArray())) 
test  = test_random.rdd.map(lambda row: LabeledPoint(row['label'], row['raw_Features'].toArray()))
 
#======== RandomForest
rf_model = RandomForest.trainClassifier(training,21,{}, 50, seed=1000) 
rf_model.totalNumNodes()#402
 
 
## Compute raw scores on the test set
# predictionAndLabels = test.map(lambda lp: (float(rf_model.predict(lp.features)), lp.label))   #doesnot work
 
# an other way 
predictions = rf_model.predict(test.map(lambda x: x.features)) 
labelsAndPredictions = test.map(lambda lp: lp.label).zip(predictions)
predictionAndLabels=labelsAndPredictions.map(lambda x: (x[1],x[0] ))  #infact , there is no need  to switch

metrics = MulticlassMetrics(predictionAndLabels) 
#metrics = MulticlassMetrics(labelsAndPredictions)  # the same as above

#predictionAndLabels – an RDD of (prediction, label) pairs.

 
#Overall statistics
print("Recall = %s" % metrics.recall())
print("Precision = %s" % metrics.precision())
print("F1 measure = %s" % metrics.fMeasure())
print("Accuracy = %s" % metrics.accuracy)
 

# # Weighted stats
print("Weighted recall = %s" % metrics.weightedRecall)
print("Weighted precision = %s" % metrics.weightedPrecision)
print("Weighted F1 Score = %s" % metrics.weightedFMeasure())
 
# Recall = 0.5698258763977951
# Precision = 0.5698258763977951
# F1 measure = 0.5698258763977951
# Accuracy = 0.5698258763977951
# Weighted recall = 0.5698258763977951
# Weighted precision = 0.3707414003927403
# Weighted F1 Score = 0.4233087352774004

 
# # Statistics by class
labels = test.map(lambda lp: lp.label).distinct().collect()
for label in sorted(labels):
    print (label)   
    print("Class %s precision = %s" % (label, metrics.precision(label)))
    print("Class %s recall = %s" % (label, metrics.recall(label)))
    print("Class %s F1 Measure = %s" % (label, metrics.fMeasure(label, beta=1.0)))

# 0.0
# Class 0.0 precision = 0.5709591342590304
# Class 0.0 recall = 0.9937217802623092
# Class 0.0 F1 Measure = 0.7252271336440238
# 1.0
# Class 1.0 precision = 0.49554655870445347
# Class 1.0 recall = 0.07749778396859566
# Class 1.0 F1 Measure = 0.13403416557161626
# 2.0
# Class 2.0 precision = 0.0
# Class 2.0 recall = 0.0
# Class 2.0 F1 Measure = 0.0
# 3.0
# Class 3.0 precision = 0.0
# Class 3.0 recall = 0.0
# Class 3.0 F1 Measure = 0.0
# 4.0
# Class 4.0 precision = 0.0
# Class 4.0 recall = 0.0
# Class 4.0 F1 Measure = 0.0
# 5.0
# Class 5.0 precision = 0.0
# Class 5.0 recall = 0.0
# Class 5.0 F1 Measure = 0.0
# 6.0
# Class 6.0 precision = 0.0
# Class 6.0 recall = 0.0
# Class 6.0 F1 Measure = 0.0
# 7.0
# Class 7.0 precision = 0.0
# Class 7.0 recall = 0.0
# Class 7.0 F1 Measure = 0.0
# 8.0
# Class 8.0 precision = 0.0
# Class 8.0 recall = 0.0
# Class 8.0 F1 Measure = 0.0
# 9.0
# Class 9.0 precision = 0.0
# Class 9.0 recall = 0.0
# Class 9.0 F1 Measure = 0.0
# 10.0
# Class 10.0 precision = 0.0
# Class 10.0 recall = 0.0
# Class 10.0 F1 Measure = 0.0
# 11.0
# Class 11.0 precision = 0.0
# Class 11.0 recall = 0.0
# Class 11.0 F1 Measure = 0.0
# 12.0
# Class 12.0 precision = 0.0
# Class 12.0 recall = 0.0
# Class 12.0 F1 Measure = 0.0
# 13.0
# Class 13.0 precision = 0.0
# Class 13.0 recall = 0.0
# Class 13.0 F1 Measure = 0.0
# 14.0
# Class 14.0 precision = 0.0
# Class 14.0 recall = 0.0
# Class 14.0 F1 Measure = 0.0
# 15.0
# Class 15.0 precision = 0.0
# Class 15.0 recall = 0.0
# Class 15.0 F1 Measure = 0.0
# 16.0
# Class 16.0 precision = 0.0
# Class 16.0 recall = 0.0
# Class 16.0 F1 Measure = 0.0
# 17.0
# Class 17.0 precision = 0.0
# Class 17.0 recall = 0.0
# Class 17.0 F1 Measure = 0.0
# 18.0
# Class 18.0 precision = 0.0
# Class 18.0 recall = 0.0
# Class 18.0 F1 Measure = 0.0
# 19.0
# Class 19.0 precision = 0.0
# Class 19.0 recall = 0.0
# Class 19.0 F1 Measure = 0.0
# 20.0
# Class 20.0 precision = 0.0
# Class 20.0 recall = 0.0
# Class 20.0 F1 Measure = 0.0
