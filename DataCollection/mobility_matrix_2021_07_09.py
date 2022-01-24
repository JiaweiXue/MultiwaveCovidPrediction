#!/usr/bin/env python
# coding: utf-8

# In[1]:


#This code generates the embeddings for the mobility.
#period: March 1, 2021 to July 5, 2021. ~125days.
#1. mobility data:
#the number of mobility data: ~50000K 
#the number of mobility data: ~50 points per day per user
#total number of point data: per day. 25M points.
#5 min per day =  12 * 33 = 396 files. 

#2. text data:
#the number of daily texted users: ~200K
#the number of daily texted records: ~3M, ~15 search per users
#5 min per day = 12 * 33 = 396 files.


# In[2]:


import os
import csv
import json 
import time
import warnings
import datetime
import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point
warnings.filterwarnings('ignore')


# In[3]:


##codes to obtain the mobility feature
##2021.07.09
##author Jiawei 
##param args


# # This code is used for obtaining mobility feature X based on the mobility.csv every day

# # function1: convert the trajectory data to dict_gps file 

# In[4]:


#input: the file location of the gps record of all users on one day 
##each row: [id,date,hour,min,second,lon,lat]
#output: the dictionary of the gps records
##dict: {"id1":{"second1",[lon1,lat1],...}...}
def csv_to_dict_gps(input_csv):
    output_dict = {}
    num_record = 0  #20210503
    with open(input_csv) as f:   #read the csv 
        reader = csv.reader(f)   #define the reader
        for row in reader:       #read each row
            if row is not None:
                row_data = row
                if (len(row_data)>=7):            #20210503
                    num_record = num_record + 1   #20210503
                    user_id, lon1, lat1 = row_data[0], float(row_data[5]), float(row_data[6])
                    seconds = int(row_data[2])*3600 + int(row_data[3])*60 + int(row_data[4])  #calculate the seconds from 0:00:00
                    if user_id not in output_dict.keys():
                        output_dict[user_id] = {}
                    output_dict[user_id][str(seconds)] = [lon1, lat1]   #the user is at longitude1, latitude1 at time seconds
    print ("total number of valid mobility records on this day: ", num_record)
    return output_dict


# # function2: convert the dict_gps to dict_zone

# In[5]:


#input: the gps points for one user: {"second1":[lon1,lat1],...}; "second" is the string of seconds starting from 0:00:00
#output: ["seconds1","seconds2",...]; sort the seconds from smallest to largest
def get_order_time(data_dict):
    key_list = list(data_dict.keys())
    key_list = [int(key_list[i]) for i in range(len(key_list))]
    key_list.sort()   #sort the seconds
    output_list = [str(key_list[i]) for i in range(len(key_list))]
    return output_list


# In[6]:


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
    return (join_result)


# In[7]:


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


# In[8]:


#input1: d_json, {"id1":{"seconds":[lon,lat],...},...}; the collection of gps data for all users on one day
#input2: shapefile; shapefile of Tokyo
#output: output, {zone1_zone2:,number}; x_12 represents the flow from zone1 to zone2 on one day
def dict_gps_to_dict_zone(d_json,grid_mapping_result,tokyo_sf):
    print ("the number of users on this day", len(d_json))
    jcode_list = tokyo_sf["JCODE"]
    mobility_dict = {str(jcode_list[i])+'_'+str(jcode_list[j]):0 for i in range(len(jcode_list)) for j in range(len(jcode_list))}
    num = 0
    for key in d_json:                                     #for loop on all users
        num+=1
        if num%1000==0:
            print ("the number of users that have been read", num)
        record = d_json[key]                               #the record for this user
        seconds_order = get_order_time(record)             #get the time order for the records of this user
        n_point = len(seconds_order)                       #number of gps points for this user
        if n_point >= 2:
            record_list = [record[seconds_order[i]] for i in range(len(seconds_order))]   
            zone_id_list = point_to_zone_2(record_list, grid_mapping_result) #['12115','0',...]
            for i in range(len(zone_id_list)-1):
                from_idx, to_idx = zone_id_list[i], zone_id_list[i+1]
                if len(from_idx)>1 and len(to_idx)>1 and i<len(seconds_order)-1:
                    if int(seconds_order[i+1])-int(seconds_order[i])>600: 
                        #20210520: the two points have the gap of larger than 15 mins.
			#20210530: the two points have the gap of larger than 10 mins.
                        mobility_dict[str(from_idx)+'_'+str(to_idx)] +=1
    return mobility_dict


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


#step3
folder_traj = "/mnt/jiawei/mobility"  
###attention: please change to the location of csv file generated by "covid19_prediction_mobility.java"

begin = datetime.date(2021,3,1)   #the start date
end = datetime.date(2021,7,5)    #the end date
d = begin 
delta = datetime.timedelta(days=1)
d_user_point = dict()
while d <= end:                           #make the iteration on days
    str_date = d.strftime("%Y%m%d")
    print ("this day is", str_date) #20210503
    #read the gps trajectory data, csv
    file_traj = str_date + ".csv"                                                
    ###attention:please change to the name of csv file generated by "covid19_prediction_mobility.java"
    
    path_traj = os.path.join(folder_traj, file_traj) 
    #d_traj = pd.read_csv(path_traj)  
    try:
        #function1: convert the trajectory data to dict_gps file 
        print("start to read mobility data")   #20210503
        d_dict_gps = csv_to_dict_gps(path_traj) 
        print("mobility "+str_date + " file found!")
        
        #function2: convert the dict_gps to dict_zone
        d_zone = dict_gps_to_dict_zone(d_dict_gps,grid_mapping_result,tokyo_sf) 
        print ("successfully map the mobility data to zones")             #20210503
        
        #write the mobility matrix for this day
        output_file_name =  str_date
        with open("mobility_feature_" + output_file_name + ".json", "w") as outfile:  
            json.dump(d_zone, outfile) 
        print ("successfully generate the mobility embedding on this day") #20210503
           
        #20210520
        num_user, num_point = len(d_dict_gps), 0
        for item in d_dict_gps:
            num_point += len(d_dict_gps[item])
        print ("the number of users on this day: ", num_user)
        print ("the number of points on this day: ", num_point)
        d_user_point[str_date] = {"num_user":num_user, "num_point":num_point}
      
    except FileNotFoundError:
        print("mobility " +str_date+" file not found!")
    d += delta                                           #update the date
with open("mobility_user_point.json", "w") as outfile2:  
    json.dump(d_user_point, outfile2)


# In[ ]:





# In[ ]:




