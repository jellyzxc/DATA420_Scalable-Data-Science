#---------Q1--------------
#check the data sets
hdfs dfs -ls /data/ghcnd/
hdfs dfs -ls /data/ghcnd/daily

##===show  directory tree
hadoop dfs -lsr /data/ghcnd/ | awk '{print $8}' | \sed -e 's/[^-][^\/]*\//--/g' -e 's/^/ /' -e 's/-/|/'

#how many files  in /data/ghcnd/daily  
hdfs dfs -ls /data/ghcnd/daily |wc -l
#259-1=258
 
##===get data size 
#total o
hadoop fs -du  -s -h /data/ghcnd/   

#seperate
hadoop fs -du  -h /data/ghcnd/

#get the files' information(including file name and size) and save in the tx file ,then load to a excel to draw a plot
hdfs dfs -ls /data/ghcnd/daily > daily.txt



#---------Q2 Q3--------------

# 2executors, 1 core per executor, 1 GB of executor memory, and 1 GB  of master memory

#look at the data
hdfs dfs -head /data/ghcnd/countries  
hdfs dfs -head /data/ghcnd/inventory  
hdfs dfs -head /data/ghcnd/states  
hdfs dfs -head /data/ghcnd/stations  
hdfs dfs -cat /data/ghcnd/daily/2020.csv.gz | gunzip | head -n 5

#create a output directory
hdfs dfs -mkdir -p Assign1/output


#start_pyspark_shell -e 2 -c 1 -w 1 -m 1 
submit_pyspark_script Assignmnet1_Processing_Q2Q3.py -e 2 -c 1 -w 1 -m 1


hadoop fs -du  -h /user/xzh216/Assign1/output

 
 
 
 
