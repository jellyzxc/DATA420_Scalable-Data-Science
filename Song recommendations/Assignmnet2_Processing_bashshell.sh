#---------Q1--------------
#check the data sets
hdfs dfs -ls /data/msd/
hdfs dfs -ls /data/msd/main/summary
hdfs dfs -ls /data/msd/tasteprofile/mismatches 
hdfs dfs -ls /data/msd/tasteprofile/triplets.tsv
hdfs dfs -ls /data/msd/audio/features/ 

##===show  directory tree
hadoop dfs -lsr /data/msd/ | awk '{print $8}' | \sed -e 's/[^-][^\/]*\//--/g' -e 's/^/ /' -e 's/-/|/'
 
  
##===get data size 
#total o
hadoop fs -du  -s -h /data/msd/   

#seperate
hadoop fs -du  -h /data/msd/

hadoop fs -du  -h /data/msd/tasteprofile/
hadoop fs -du  -h /data/msd/tasteprofile/mismatches
hadoop fs -du  -h /data/msd/tasteprofile/triplets.tsv

hadoop fs -du  -h /data/msd/audio/
hadoop fs -du  -h /data/msd/audio/attributes
hadoop fs -du  -h /data/msd/audio/features 
hadoop fs -du  -h /data/msd/audio/statistics  

hadoop fs -du  -h /data/msd/genre
 
hadoop fs -du  -h /data/msd/main/summary/

##===look at the data
hdfs dfs -cat /data/msd/main/summary/analysis.csv.gz | gunzip | head -n 5
hdfs dfs -cat /data/msd/main/summary/metadata.csv.gz | gunzip | head -n 5
 
hdfs dfs -head /data/msd/tasteprofile/mismatches/sid_matches_manually_accepted.txt
hdfs dfs -head /data/msd/tasteprofile/mismatches/sid_mismatches.txt


# msd-rp-v1.0.csv 
# msd-ssd-v1.0.csv 
# msd-rh-v1.0.csv 
# msd-tssd-v1.0.csv 
# msd-trh-v1.0.csv 
# msd-mvd-v1.0.csv 	
# msd-marsyas-timbral-v1.0.csv 
# msd-jmir-spectral-all-all-v1.0.csv 
# msd-jmir-spectral-derivatives-all-all-v1.0.csv 
# msd-jmir-methods-of-moments-all-v1.0.csv 
# msd-jmir-area_of_moments-all-v1.0.csv 
# msd-jmir-lpc-all-v1.0.csv 
# msd-jmir-mfcc-all-v1.0.csv 
 
hdfs dfs -cat /data/msd/tasteprofile/triplets.tsv/part-00000.tsv.gz | gunzip | head -n 5
hdfs dfs -cat /data/msd/audio/attributes/msd-tssd-v1.0.attributes.csv | head -n 10
#"component_1",NUMERIC

hdfs dfs -cat /data/msd/audio/attributes/msd-jmir-area-of-moments-all-v1.0.attributes.csv | head -n 10
#Area_Method_of_Moments_Overall_Standard_Deviation_1,real

hdfs dfs -cat /data/msd/audio/attributes/msd-mvd-v1.0.attributes.csv | head -n 10
#"component_0",NUMERIC

hdfs dfs -cat /data/msd/audio/attributes/msd-rp-v1.0.attributes.csv | head -n 10
#"component_1",NUMERIC

hdfs dfs -cat /data/msd/audio/attributes/msd-marsyas-timbral-v1.0.attributes.csv | head -n 5
#Mean_Acc5_Mean_Mem20_Centroid_Power_powerFFT_WinHamming_HopSize512_WinSize512_Sum_AudioCh0,real

hdfs dfs -cat /data/msd/audio/features/msd-marsyas-timbral-v1.0.csv/part-* | gunzip | head -n 5  

hdfs dfs -cat /data/msd/audio/features/msd-jmir-area-of-moments-all-v1.0.csv/part-* | gunzip | head -n 5
hdfs dfs -cat /data/msd/audio/features/msd-marsyas-timbral-v1.0.csv/part-* | gunzip | head -n 5


hdfs dfs -cat /data/msd/genre/msd-MAGD-genreAssignment.tsv | head -n 5  



##===row couts in data set ====wc - l   all counts need to -1

#== 1 main/summary
hdfs dfs -cat /data/msd/main/summary/metadata.csv.gz | gunzip | wc -l
#1000001
hdfs dfs -cat /data/msd/main/summary/analysis.csv.gz | gunzip | wc -l
#1000001


#==2  audio

#attributes   eg
hadoop fs -du  -h /data/msd/audio/attributes
hdfs dfs -cat /data/msd/audio/attributes/msd-tssd-v1.0.attributes.csv | wc -l
#1177

hdfs dfs -cat /data/msd/audio/attributes/msd-marsyas-timbral-v1.0.attributes.csv | wc -l
#125

hdfs dfs -cat /data/msd/audio/attributes/msd-jmir-spectral-derivatives-all-all-v1.0.attributes.csv | wc -l
#17

#features   eg 
hadoop fs -du  -h /data/msd/audio/features
hdfs dfs -cat /data/msd/audio/features/msd-tssd-v1.0.csv/part-* | gunzip | wc -l
#994188

hdfs dfs -cat /data/msd/audio/features/msd-jmir-area-of-moments-all-v1.0.csv/part-* | gunzip | wc -l
#994623

hdfs dfs -cat /data/msd/audio/features/msd-marsyas-timbral-v1.0.csv/part-* | gunzip | wc -l
#995001

hdfs dfs -cat /data/msd/audio/features/msd-jmir-spectral-derivatives-all-all-v1.0.csv/part-* | gunzip | wc -l
#994623



#statistics  
hadoop fs -du  -h /data/msd/audio/statistics
hdfs dfs -cat /data/msd/audio/statistics/sample_properties.csv.gz | gunzip | wc -l
#992866


#==3 tasteprofile
#==tasteprofile/mismatches 
hdfs dfs -cat /data/msd/tasteprofile/mismatches/sid_matches_manually_accepted.txt | wc -l
#938
hdfs dfs -cat /data/msd/tasteprofile/mismatches/sid_mismatches.txt | wc -l
#19094     

#==tasteprofile/triplets.tsv 
hdfs dfs -cat /data/msd/tasteprofile/triplets.tsv/part-* | gunzip | wc -l
#48373586

 
#==3 genre 
hdfs dfs -cat /data/msd/genre/msd-MAGD-genreAssignment.tsv | wc -l
#422714
hdfs dfs -cat /data/msd/genre/msd-MASD-styleAssignment.tsv | wc -l
#273936
hdfs dfs -cat /data/msd/genre/msd-topMAGD-genreAssignment.tsv | wc -l
#406427


#create file to save the result
hdfs dfs -mkdir -p Assign2/output

hdfs dfs -ls /user/xzh216/Assign2/output

hadoop fs -du  -h /user/xzh216/Assign2/output