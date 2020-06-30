#increase your resources up to 4 executors, 2 cores per executor, 4 GB of executor memory, and 4 GB of master memory.
# start_pyspark_shell -e 4 -c 2 -w 4 -m 4 
 

#---------Q1  Q2-----------------------
hdfs dfs -ls /user/xzh216/Assign1/output
 
submit_pyspark_script Assignmnet1_Analysis_Q1Q2.py -e 4 -c 2 -w 4 -m 4 

# check the results in  hdfs  countries ,states  and  StationsDistance_NZ
hdfs dfs -du  -h /user/xzh216/Assign1/output



#---------Q3 ----------------------
hdfs getconf -confKey "dfs.blocksize"
#134217728   =>128M


hdfs dfs -du  -h /data/ghcnd/daily/2020.csv.gz
# 30.2 M    <128M     


hdfs dfs -du  -h /data/ghcnd/daily/2010.csv.gz
#221.3 M    2 Blocks


# eg.  hdfs fsck /data/helloworld/part-i-00000 -files -blocks -locations
hdfs fsck /data/ghcnd/daily/2010.csv.gz -blocks  
# 110.6646528M/block


hdfs fsck /data/ghcnd/daily/2015.csv.gz -blocks  

#Total size:    207618101 B
#Total files:   1
#Total blocks (validated):      2 (avg. block size 103809050 B)


hdfs fsck /data/ghcnd/daily/2020.csv.gz -blocks   
# 31626590 B  =   30.2 M     1block



submit_pyspark_script Assignmnet1_Analysis_Q3.py -e 4 -c 2 -w 4 -m 4 

# submit_pyspark_script SCRIPT [-e executors] [-c cores] [-w worker memory] [-m master memory] [-n name]
#--------------------Q4----------------------
#start_pyspark_shell -e 4 -c 4 -w 4 -m 4
#start_pyspark_shell -e 8 -c 4 -w 8 -m 4   #32    
 

hdfs fsck /data/ghcnd/daily -blocks 
# Replicated Blocks:
# Total size:    16639100391 B
# Total files:   258
#Total blocks (validated):      327 (avg. block size 50884099 B)
 
#Permission denied: '/usr/local/anaconda3/lib/python3.7/site-packages/docopt.py'
# pip install hdfs  
pip install hdfs --target=/users/home/xzh216/pythonPackage
pip install hdfs3 --target=/users/home/xzh216/pythonPackage

 

submit_pyspark_script Assignmnet1_Analysis_Q4.py -e 8 -c 4 -w 4 -m 4 

 
# check all  the  output  file  
hdfs dfs -ls /user/xzh216/Assign1/output

hdfs dfs -stat /user/xzh216/Assign1/output/daily_with_tempelements_NZ.orc 
hdfs fsck /user/xzh216/Assign1/output/daily_with_tempelements_NZ.orc -blocks    

hdfs fsck /user/xzh216/Assign1/output/daily_Avg_Rainfall.orc -blocks    
 
#copy the output   daily_with_tempelements_NZ    from HDFS to  local  directory 
hdfs dfs  -ls /user/xzh216/Assign1/output/daily_with_tempelements_NZ.csv
 

hdfs dfs -copyToLocal  /user/xzh216/Assign1/output/daily_with_tempelements_NZ.csv  /users/home/xzh216/Assign1/Result
 
ll /users/home/xzh216/Assign1/Result/daily_with_tempelements_NZ.csv

wc -l /users/home/xzh216/Assign1/Result/daily_with_tempelements_NZ.csv/part-00000-00127051-eee8-4454-a64c-507e7478a397-c000.csv
#458892

hdfs dfs -copyToLocal  /user/xzh216/Assign1/output/daily_Avg_Rainfall.csv  /users/home/xzh216/Assign1/Result
hdfs dfs -copyToLocal  /user/xzh216/Assign1/output/daily_Cum_Rainfall.csv  /users/home/xzh216/Assign1/Result



#--------------------Challenge----------------------
submit_pyspark_script Assignmnet1_Chanllenge.py -e 8 -c 4 -w 4 -m 4 
