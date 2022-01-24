#!/usr/bin/env python
# coding: utf-8

# In[1]:


#This code generates the embeddings for the text.
#period: March 1, 2021 to July 5, 2021. ~125 days.
#2. text data:
#the number of daily texted users: ~200K
#the number of daily texted records: ~3M, ~15 searches per users


# In[2]:


import os
import csv
import time
import json 
import warnings
import datetime
import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point
warnings.filterwarnings('ignore')


# In[3]:


##codes to obtain the web search embedding
##2021.07.09
##author: Jiawei 


# In[4]:


#function 1
#input1: [[lon1,lat1],[lon2,lat2],[lon3,lat3],...]
#input2: shapefile
#output: {(139.663, 35.692): '13115',...}
def point_to_zone_1(data_point,shapefile):
    dp_num = len(data_point)
    h = pd.DataFrame({'zip':[i for i in range(dp_num)],                      'Lat':[data_point[i][1] for i in range(dp_num)],                      'Lon':[data_point[i][0] for i in range(dp_num)]})
    crs ='epsg:4612'    #the same as the crs of Tokyo shapefile
    geometry = [Point(xy) for xy in zip(h.Lon, h.Lat)]
    hg = gpd.GeoDataFrame(h, crs=crs, geometry=geometry)  #define the point
    join = gpd.sjoin(hg, shapefile, how="inner", op="within")  #find the region where the point is located
    print ("the number of grid points within Tokyo:", len(join))
    geo_list, jcode_list = list(join["geometry"]), list(join["JCODE"])
    join_result = {(round(geo_list[i].x,3), round(geo_list[i].y,3)):str(jcode_list[i]) for i in range(len(join))}
    return join_result

#function 2
#input: [[lon1, lat1],[lon2,lat2],[lon3,lat3],...]
#output:['12115','0',...]
def point_to_zone_2(point_list, grid_result):
    result = list()
    for i in range(len(point_list)):
        lon, lat = point_list[i][0], point_list[i][1]
        lon_grid, lat_grid = round(lon,3),round(lat,3)
        if (lon_grid, lat_grid) in grid_result:
            result.append(grid_result[(lon_grid, lat_grid)])
        else:
            result.append('0')
    return result

#function 3: find the zone location of each user id
#input: the id_homelocs.csv path; grid_mapping_result
#output: a dict, {"idx": "zoneid",...}
def id_to_zone(input_csv,grid_mapping_result):
    output_dict = {}
    number_user = 0              #20210503
    with open(input_csv) as f:   #read the csv
        reader = csv.reader(f)   #define the reader
        for row in reader:       #read each row
            if row is not None:    
                row_data = row
                if (len(row_data)>=3):   #20210503
                    number_user += 1   #20210503
                    idx,lon,lat = row_data[0], float(row_data[1]), float(row_data[2])
                    zone_id_result = point_to_zone_2([[lon,lat]], grid_mapping_result)
                    zone_id = zone_id_result[0]
                    output_dict[idx] = zone_id
    print ("the total number of users is ", number_user)  #20210503
    return output_dict

#function 4: aggregate the frequency 
#input1: frequency for each id, 
#each row: id,"fever","1","chills","2",...
#input2: id_zone
#{"id":"zone id"}
#output: the frequency for each zone
#{"zone_id":{"fever":1,"chills":2,...},...}
def aggregate_text(input_csv,id_zone,tokyo_sf):
    print ("the total number of rows on this day:", len(input_csv))
    jcode_list = tokyo_sf["JCODE"]
    text_dict = {jcode_list[i]:{} for i in range(len(jcode_list))}
    text_record = 0 #20210503
    num  = 0
    with open(input_csv) as f:   #read the csv
        reader = csv.reader(f)   #define the reader
        for row in reader:       #read each row
            num = num + 1
            if num%1000==0:
                print ("the number of rows that have been read", num)
            if row is not None:    
                row_data = row
                if len(row_data)>=1: #20210503
                    idx = row_data[0]               #get the id of this user
                    idx_zone = id_zone[idx]         #get the zone of this user
                    if idx_zone in text_dict.keys() and len(row_data)%2==1:  #the zone is within Tokyo
                        num_word = int((len(row_data)-1)/2) #number of words by this user
                        if num_word > 0:
                            text_record += 1           #20210503
                            for j in range(num_word):
                                word, count = row_data[2*j+1], int(row_data[2*j+2])  
                                if word not in text_dict[idx_zone].keys():
                                    text_dict[idx_zone][word] = count
                                else:
                                    text_dict[idx_zone][word] += count
    print ("total number of valid text records", text_record) #20210503
    return text_dict


# # main function

# In[9]:


#step1: read the Tokyo shapefile   
folder_sf = "/mnt/jiawei/code_from_jiawei_to_taka_kota_20210503/tokyo_shapefile"  #attention!!!
#folder_sf = "/data/HSEES/xue_codes/cikm_prediction/tokyo_shapefile"

file_sf = "tokyo.shp"                                                     
path_sf = os.path.join(folder_sf,file_sf)  
tokyo_sf = gpd.read_file(path_sf)
print ("successfully load the shapefile") #20210503
print ("the total number of zones is") #20210503
print (len(tokyo_sf))


# In[10]:


#step 2: build the grid network
gap = 0.001
lon_min, lon_max = 139.55, 139.92
lon_num = round((lon_max - lon_min)/gap) + 1
lat_min, lat_max = 35.52, 35.82
lat_num = round((lat_max - lat_min)/gap) + 1
lon_lat_list = list()
for i in range(lon_num):
    for j in range(lat_num):
        lon, lat = round(lon_min + i * gap,3), round(lat_min + j * gap,3)
        lon_lat_list.append([lon, lat])
#step 2: find the located zones of all points in the grid network
print ("expected waiting time to map all grid points: less than 1 min")
time1 = time.time()
grid_mapping_result = point_to_zone_1(lon_lat_list, tokyo_sf)
time2 = time.time()
print ("total time until now", time2 - time1)
print ("the zone of point (139.65, 35.70):", grid_mapping_result[(139.65,35.70)])


# In[11]:


#step 3. read the id_home
folder_id_home = "/mnt/jiawei/mobility" 
###attention: please change to location of "id_homelocs.csv", which is the output of code "covid19_prediction_home.java"

file_id_home = "id_homelocs.csv"
path_id_home = os.path.join(folder_id_home,file_id_home)


# In[12]:


#step 4. the raw web search record file
folder_text = "/mnt/jiawei/text"  
###attention: change to location of "home_text+date_str+"_text"+".csv", which is the output of "covid19_prediction_text.java"

begin = datetime.date(2021,3,1)   #the start date
end = datetime.date(2021,7,5)    #the end date
d = begin 
delta = datetime.timedelta(days=1)
id_zone = id_to_zone(path_id_home,grid_mapping_result)  #call function 2
while d <= end:                              #make the iteration on days
    str_date = d.strftime("%Y%m%d")
    print ("this day is", str_date) #20210503
    #read the text file
    file_text = str_date + "_text" + ".csv"           
    path_text = os.path.join(folder_text, file_text)  
    try:        
        d_text = aggregate_text(path_text, id_zone, tokyo_sf)  #call function 3
        print ("finish aggregating the text data on this day") #20210503
        output_file_name =  str_date
        with open("text_feature_" + output_file_name + ".json", "w") as outfile:  
            json.dump(d_text, outfile) 
            print(str_date+" file found!")
    except FileNotFoundError:
        print(str_date+" file not found!")
    d += delta      


# In[ ]:





# In[ ]:




