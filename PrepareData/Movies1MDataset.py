'''
Created on 23.09.2016

@author: hp
'''

import os
import pandas
import numpy
import datetime
import pytz
import re
import csv
import time
import sys


class Consts(object):
    def __init__(self):
        self.data_path = "F:\\works\\projects\\python\\RecommenderSystem\\data"
        self.timezone_path = "F:\\works\\projects\\python\\RecommenderSystem\\zipcode.csv"
        
        self.movies_path = os.path.join(self.data_path,"pre_movies.csv")
        self.users_path = os.path.join(self.data_path,"pre_users.csv")
        self.ratings_path = os.path.join(self.data_path,"pre_ratings.csv")
        self.users_csv_file_name = os.path.join(self.data_path,"users.csv")
        self.movies_csv_file_name = os.path.join(self.data_path,"movies.csv")
        self.ratings_csv_file_name = os.path.join(self.data_path,"ratings.csv")
        self.min_year = 1900
        
        self.users_cvs_file_name = os.path.join(self.data_path,"users.csv")
        self.movies_cvs_file_name = os.path.join(self.data_path,"movies.csv")
        self.ratings_cvs_file_name = os.path.join(self.data_path,"ratings.csv") 
        self.userids_npy_file_name = os.path.join(self.data_path,"userids.npy")
        self.moviesids_npy_file_name = os.path.join(self.data_path,"moviesids.npy")
        self.ratings_by_user_npy_file_name = os.path.join(self.data_path,"ratings_by_user.npy")
        self.ratings_by_user_ids_npy_file_name = os.path.join(self.data_path,"ratings_by_user_ids.npy")
        self.ratings_by_user_idx_npy_file_name = os.path.join(self.data_path,"ratings_by_user_idx.npy")
        self.ratings_by_movie_npy_file_name = os.path.join(self.data_path,"ratings_by_movie.npy") 
        self.ratings_by_movie_ids_npy_file_name = os.path.join(self.data_path,"ratings_by_movie_ids.npy") 
        self.ratings_by_movie_idx_npy_file_name = os.path.join(self.data_path,"ratings_by_movie_idx.npy")
        
        self.MaxRate = 5
        return
    pass 


def convert_csv(consts):
    print("converting csv...")
    timezones_cvs = pandas.read_csv(
        consts.timezone_path
        ,dtype = {
            'zip':numpy.str
            ,'city':numpy.str
            ,'state':numpy.str
            ,'latitude':numpy.float32
            ,'longitude':numpy.float32
            ,'timezone':numpy.int32
            ,'dst':numpy.int32
            }
        ,index_col = False
        )
    print("timezone data was loaded")
    movies_cvs = pandas.read_csv(
        consts.movies_path
        ,sep=";"
        ,header=None
        ,quotechar='"'
        ,encoding="cp1251"
        ,names=("MovieID","Name","Genders")
        ,dtype = {
            'MovieID':numpy.int32
            ,'Name':numpy.str
            ,'Genders':numpy.str
            }
        ,index_col = False
        )
    print("movies data was loaded")
    users_cvs = pandas.read_csv(
        consts.users_path
        ,sep=";"
        ,header=None
        ,quotechar='"'
        ,encoding="cp1251"
        ,names=("UserID","Gender","Age","Occupation","ZipCode")
        ,dtype = {
            'UserID':numpy.int32
            ,'Gender':numpy.str
            ,'Age':numpy.int32
            ,'Occupation':numpy.int32
            ,"ZipCode":numpy.str
            }
        ,index_col = False
        )
    print("users data was loaded")
    ratings_cvs = pandas.read_csv(
        consts.ratings_path
        ,sep=";"
        ,header=None
        ,quotechar='"'
        ,encoding="cp1251"
        ,names=("UserID","MovieID","Rating","Timestamp")
        ,dtype = {
            'UserID':numpy.int32
            ,'MovieID':numpy.int32
            ,'Rating':numpy.float32
            ,'Timestamp':numpy.int32
            }
        ,index_col = False
        )
    print("ratings data was loaded")
    
    lt = time.time()
    prog = re.compile(pattern = "\((\d+)\)$")
    movies_cvs['year'] = int(consts.min_year)
    for i in numpy.arange(movies_cvs.shape[0]-1):
        name = str(movies_cvs.at[i,"Name"])
        m = prog.search(name)
        if m:
            movies_cvs.at[i,'year'] = int(m.group(1))
            pass
        t1 = time.time()
        if t1>lt+1:
            p = float(i)/float(movies_cvs.shape[0])*100.0
            sys.stdout.write("\t\t\t\t\t\t\t\t\t\r")
            sys.stdout.write("movies csv data process %f %%\r" % (p,))
            lt = lt+1
        pass
    print("movies cvs data was prepared")
    
    users_cvs['latitude'] = float(0)
    users_cvs['longitude'] = float(0)
    users_cvs['timezone'] = int(0)
    users_cvs['dts'] = int(0)
    for i in numpy.arange(users_cvs.shape[0]-1):
        zipcode = users_cvs.loc[i,'ZipCode']
        zc = timezones_cvs[timezones_cvs.zip.isin([zipcode])]
        if len(zc)==1:
            users_cvs.at[i,'timezone'] = int(zc['timezone'])
            users_cvs.at[i,'latitude'] = float(zc['latitude'])
            users_cvs.at[i,'longitude'] = float(zc['longitude'])
            users_cvs.at[i,'dts'] = int(zc['dst'])
            pass  
        t1 = time.time()
        if t1>lt+1:
            p = float(i)/float(users_cvs.shape[0])*100.0
            sys.stdout.write("\t\t\t\t\t\t\t\t\t\r")
            sys.stdout.write("users csv data process %f %%\r" % (p,))
            lt = lt+1
        pass
    print("users cvs data was prepared")
    
    ratings_cvs["wday"] = int(0)
    ratings_cvs["yday"] = int(0)
    ratings_cvs["year"] = int(consts.min_year)
    
    for i in numpy.arange(ratings_cvs.shape[0]-1):
        user_id = int(ratings_cvs.at[i,"UserID"])
        t0 = ratings_cvs.at[i,"Timestamp"]
        ui = users_cvs[users_cvs.UserID.isin([user_id])]
        if len(ui)==1:
            timezone = int(ui["timezone"]) - 2
            tt = datetime.datetime.fromtimestamp(t0,datetime.timezone(datetime.timedelta(hours=timezone))).timetuple()
            ratings_cvs.at[i,"wday"] = tt.tm_wday
            ratings_cvs.at[i,"yday"] = tt.tm_yday
            ratings_cvs.at[i,"year"] = tt.tm_year 
            pass
        t1 = time.time()
        if t1>lt+1:
            p = float(i)/float(ratings_cvs.shape[0])*100.0
            sys.stdout.write("\t\t\t\t\t\t\t\t\t\r")
            sys.stdout.write("ratings csv data process %f %%\r" % (p,))
            lt = lt+1
        pass
    print("ratings cvs data was prepared")
    
    users_cvs.to_csv(
        path_or_buf = consts.users_csv_file_name, 
        sep = ";"
        ,header = False
        ,index = False
        ,encoding = "utf-8"
        ,quoting = csv.QUOTE_ALL
        ,quotechar = '"'
        ,line_terminator = "\n"
        ,doublequote = True
        )
    movies_cvs.to_csv(
        path_or_buf = consts.movies_csv_file_name, 
        sep = ";"
        ,header = False
        ,index = False
        ,encoding = "utf-8"
        ,quoting = csv.QUOTE_ALL
        ,quotechar = '"'
        ,line_terminator = "\n"
        ,doublequote = True
        )
    ratings_cvs.to_csv(
        path_or_buf = consts.ratings_csv_file_name
        ,sep = ";"
        ,header = False
        ,index = False
        ,encoding = "utf-8"
        ,quoting = csv.QUOTE_ALL
        ,quotechar = '"'
        ,line_terminator = "\n"
        ,doublequote = True
        )
    print("converting done")
    return

def get_aranged(value,min_value,max_value):
    if abs(max_value-min_value)<1e-9: 
        return 0
    return (float(value)-float(min_value))/(float(max_value)-float(min_value)) - float(0.5)        

def prepare_data(consts = Consts()):
    print("loading data...")
    
    user_indice = 0
    movie_indice = 1
    
    
    # user_cvs
    # columns: 
    #     id -- int (key); sex -- ['M'|'F']; age -- int; 
    #     occupation -- int; latitude -- real; longitude -- real; 
    #     timezone -- int; dts -- [0|1];   
    #
    users_cvs = pandas.read_csv(
        consts.users_cvs_file_name
        ,names = ("id","sex","age","occupation","zipcode","latitude","longitude","timezone","dts")
        ,dtype = {
            'id':numpy.int32
            ,'sex':numpy.str
            ,'age':numpy.int32
            ,'occupation':numpy.int32
            ,"zipcode":numpy.str
            ,'latitude':numpy.float32
            ,'longitude':numpy.float32
            ,'timezone':numpy.int32
            ,'dts':numpy.int32
            }
        ,sep=";"
        ,skipinitialspace = False
        ,header=None
        ,index_col = False
        ,quoting = csv.QUOTE_ALL
        ,quotechar='"'
        ,encoding="utf-8"
        ,na_values=''
        )
    print("The users_cvs was loaded.")
    #print(users_cvs)
    
    # movies_cvs 
    # columns:
    #    id -- int (key); name -- string; gender -- string; year -- int;
    movies_cvs = pandas.read_csv(
        consts.movies_cvs_file_name
        ,sep=";"
        ,names = ["id","name","gender","year"]
        ,dtype = {
            'id':numpy.int32
            ,'name':numpy.str
            ,'gender':numpy.str
            ,'year':numpy.int32
            }
        ,skipinitialspace = False
        ,header=None
        ,index_col = False
        ,quoting = csv.QUOTE_ALL
        ,quotechar='"'
        ,encoding="utf-8"
        )
    print("The movies_cvs was loaded.")
    #print(movies_cvs)
    
    # ratings_cvs
    # columns:
    #     userid -- int (from users_cvs id key); filmid -- int (from movies_cvs id key); 
    #     rate -- real; wday -- int; yday -- int; year -- int; 
    ratings_cvs = pandas.read_csv(
        consts.ratings_cvs_file_name
        ,sep=";"
        ,names=["userid","filmid","rate",'Timestamp',"wday","yday","year"]
        ,dtype = {
            'userid':numpy.int32
            ,'filmid':numpy.str
            ,'rate':numpy.float32
            ,'Timestamp':numpy.int32
            ,'wday':numpy.int32
            ,'yday':numpy.int32
            ,'year':numpy.int32
            }
        ,skipinitialspace = False
        ,header=None
        ,index_col = False
        ,quoting = csv.QUOTE_ALL
        ,quotechar='"'
        ,encoding="utf-8"
        )
    print("The ratings_cvs was loaded.")
    #print(ratings_cvs)
    
    
    # usersids
    # columns:
    #     sex -- +0.5 - 'M', -0.5 - 'F'
    #     age -- -0.5 - min, +0.5 - max
    
    last_user_id = users_cvs["id"][len(users_cvs)-1]
    usersids = numpy.zeros(dtype=numpy.float32,shape=(last_user_id,2))
    age_min = 1
    age_max = 56
    for i in numpy.arange(len(users_cvs)):
        if users_cvs["sex"][i]=="M":
            usersids[users_cvs["id"][i]-1,0] = 0.5
        else:
            usersids[users_cvs["id"][i]-1,0] = -0.5
        usersids[users_cvs["id"][i]-1,1] = get_aranged(value = users_cvs["age"][i], min_value = age_min, max_value = age_max)  
    print(usersids[0:100,])
    
    # moviesids 
    # columns:
    #     year -- -0.5 - min, +0.5 - max
    
    last_film_id = movies_cvs["id"][len(movies_cvs)-1]
    moviesids = numpy.zeros(dtype=numpy.float32,shape=(last_film_id,1))
    min_year = float(movies_cvs["year"].min())
    max_year = float(movies_cvs["year"].max())
    d_year = max_year - min_year
    min_year = min_year - d_year*0.1
    max_year = max_year + d_year*0.1  
    for i in numpy.arange(len(movies_cvs)):
        moviesids[movies_cvs["id"][i]-1,0] = get_aranged(value = movies_cvs["year"][i], min_value = min_year, max_value = max_year)
    print(moviesids[0:100,])

    
    ratings_cvs["id"] = numpy.arange(len(ratings_cvs))
    ratings_cvs["UserRate"] = ratings_cvs["rate"] 
    ratings_cvs["MeanRate"] = ratings_cvs["rate"] 
    grouped_by_user = ratings_cvs.groupby(by="userid")
    #mean_rate_by_user = grouped_by_user["rate"].mean()
    lt = time.time()
    i = 0
    for name,group in grouped_by_user:
        mean_rate_by_user = group["rate"].mean()
        ratings_cvs.loc[group["id"],"UserRate"] = ratings_cvs.loc[group["id"],"UserRate"] - mean_rate_by_user
        ratings_cvs.loc[group["id"],"MeanRate"] = mean_rate_by_user
        t1 = time.time()
        if t1>lt+1:
            p = float(i)/float(len(grouped_by_user))*100.0
            print("UserRates %f %%" % (p))
            lt = lt+1
        i = i + 1
    ratings_cvs["UserRate"] = ratings_cvs["UserRate"]/(2*consts.MaxRate)
    print("The UserRates column was calculated")
    print(ratings_cvs.head(100))
    
    # ratings_by_user_idx
    # columns:
    #    for one user_id, ratings_by_user_ids and ratings_by_user indexes pair
    #  
    #    start_indice -- int
    #    end_indice -- int
    
    # ratings_by_user_ids
    # columns:
    #    user_id -- int
    #    film_id -- int
    
    # ratings_by_user
    # every row for one ratings_by_user_ids row i.m. for one pair (user_id,film_id) 
    # columns:
    #    user_rate -- -0.5 - min .. +0.5 - max;
    #    wday -- -0.5 - min .. + 0.5 - max;
       
    ratings_by_user = numpy.zeros(dtype=numpy.float32,shape=(len(ratings_cvs),2))
    ratings_by_user_ids = numpy.zeros(dtype=numpy.int64,shape=(len(ratings_cvs),2))
    ratings_by_user_idx = numpy.zeros(dtype=numpy.int64,shape=(len(grouped_by_user),2))
    i = 0
    li = 0
    lt = time.time()
    j = 0
    for name,group in grouped_by_user:
        user_id = numpy.int64(name)
        for row_id in group["id"]:
            ratings_by_user_ids[j,user_indice] = user_id 
            ratings_by_user_ids[j,movie_indice] = numpy.int64(ratings_cvs.loc[row_id,"filmid"])
            ratings_by_user[j,0] = ratings_cvs.loc[row_id,"UserRate"]
            ratings_by_user[j,1] = get_aranged(value = ratings_cvs.loc[row_id,"wday"], min_value = 0, max_value = 6)
            j = j + 1
        ratings_by_user_idx[i,] = [li,li+len(group)]  
        li = li + len(group)
        t1 = time.time()
        if t1>lt+1:
            print("rating_by_user %f %%" % (float(i)/float(len(grouped_by_user))*100))
            lt = lt+1
        i = i + 1
    print("ratings_by_user rates was calculated")    
    
    # ratings_by_movie_idx
    # columns:
    #    for one movie_id, ratings_by_movie_ids and ratings_by_movie indexes pair
    #  
    #    start_indice -- int
    #    end_indice -- int
    
    # ratings_by_movie_ids
    # columns:
    #    user_id -- int
    #    film_id -- int
    
    # ratings_by_movie
    # every row for one ratings_by_movie_ids row i.m. for one pair (user_id,film_id) 
    # columns:
    #    user_rate -- -0.5 - min .. +0.5 - max;
    #    wday -- -0.5 - min .. + 0.5 - max;
       
    group_by_movie = ratings_cvs.groupby(by="filmid")       
    ratings_by_movie = numpy.zeros(dtype=numpy.float32,shape=(len(ratings_cvs),2))
    ratings_by_movie_ids = numpy.zeros(dtype=numpy.int64,shape=(len(ratings_cvs),2))
    ratings_by_movie_idx = numpy.zeros(dtype=numpy.int64,shape=(len(group_by_movie),2))
    i = 0
    li = 0
    lt = time.time()
    j = 0
    for name,group in group_by_movie:
        film_id = numpy.int64(name)
        for row_id in group["id"]:
            ratings_by_movie_ids[j,user_indice] = numpy.int64(ratings_cvs.loc[row_id,"userid"])
            ratings_by_movie_ids[j,movie_indice] = film_id 
            ratings_by_movie[j,0] = ratings_cvs.loc[row_id,"UserRate"]
            ratings_by_movie[j,1] = get_aranged(value = ratings_cvs.loc[row_id,"wday"], min_value = 0, max_value = 6)
            j = j + 1
        ratings_by_movie_idx[i,] = [li,li+len(group)]  
        li = li + len(group)
        t1 = time.time()
        if t1>lt+1:
            print("rating_by_movie %f %%" % (float(i)/float(len(group_by_movie))*100))
            lt = lt+1
        i = i + 1
    print("ratings_by_movie rates was calculated")    
    
    numpy.save(file=consts.userids_npy_file_name, arr=usersids)
    numpy.save(file=consts.moviesids_npy_file_name, arr=moviesids)
    numpy.save(file=consts.ratings_by_user_npy_file_name, arr=ratings_by_user)
    numpy.save(file=consts.ratings_by_user_ids_npy_file_name, arr=ratings_by_user_ids)
    numpy.save(file=consts.ratings_by_user_idx_npy_file_name, arr=ratings_by_user_idx)
    numpy.save(file=consts.ratings_by_movie_npy_file_name, arr=ratings_by_movie)
    numpy.save(file=consts.ratings_by_movie_ids_npy_file_name, arr=ratings_by_movie_ids)
    numpy.save(file=consts.ratings_by_movie_idx_npy_file_name, arr=ratings_by_movie_idx)    
    print("data was prepared and was saved.")
    return



if __name__ == '__main__':
    consts = Consts()
    convert_csv(consts)
    prepare_data(consts)
    pass