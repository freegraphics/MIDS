'''
Created on may 25, 2016

@author: klizardin

The MIT License (MIT)

Copyright (c) 2016 klizardin

Permission is hereby granted, free of charge, to any person obtaining a copy 
of this software and associated documentation files (the "Software"), to deal 
in the Software without restriction, including without limitation the rights 
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell 
copies of the Software, and to permit persons to whom the Software is furnished 
to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all 
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR 
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS 
FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR 
COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER 
IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN 
CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.


'''
from __future__ import print_function

import sys
import os
import time
import numpy
import pandas

from Nets import ApproxNet
from Nets import AutoEncoder
from Consts import Consts

import theano
import theano.tensor as T
from theano.tensor.shared_randomstreams import RandomStreams


def save_ids(ids,name):
    consts = Consts()
    name = os.path.join(consts.result_path,name)
    result = open(name,"wt")
    for row in numpy.arange(ids.shape[0]):
        for j in numpy.arange(ids.shape[1]):
            result.write("%f\t" % (ids[row,j],))
        result.write("\n")
    return

class IDsUpdater(object):
    '''
    The class to update ids 
    '''
    
    def __init__(self,ids_count,avg_rate,normilized_vs_avg_rate,max_distance,dist_sqrt_coef,avg_dx_item_small_weight,min_max_compresion_rate,name):
        self.ids_count = ids_count
        self.avg_rate = avg_rate
        self.normilized_vs_avg_rate = normilized_vs_avg_rate 
        self.name = name
        self.max_distance = max_distance
        self.dist_sqrt_coef = dist_sqrt_coef
        self.avg_dx_item_small_weight = avg_dx_item_small_weight
        self.min_max_compresion_rate = min_max_compresion_rate 
        return
    
    def __distance(self,ids_values,i0,i1):
        if i0==i1:
            return self.max_distance
        return numpy.sum(numpy.square(ids_values[i0,]-ids_values[i1,]))
    
    def __calc_new_ids(self,ids_values,ids_new_values,avg_rate,mask,p,item_weight):
        max_values = numpy.zeros(shape=ids_new_values.shape[1],dtype=theano.config.floatX)
        min_values = numpy.zeros(shape=ids_new_values.shape[1],dtype=theano.config.floatX)
        v1 = numpy.zeros(shape=(2,ids_new_values.shape[1]),dtype=theano.config.floatX)
        first = True
        dx_cnt = 0
        for i in numpy.arange(ids_values.shape[0]):
            if mask[i]==0:
                continue
            if first:
                max_values = ids_new_values[i,]
                min_values = ids_new_values[i,]
            else:
                v1[0,] = ids_new_values[i,]
                v1[1,] = max_values
                max_values = numpy.max(a = v1,axis = 0)
                v1[1,] = min_values
                min_values = numpy.min(a = v1,axis = 0)
            first = False
            dx_cnt = dx_cnt + 1
              
        ids_dx = numpy.zeros(shape=ids_values.shape,dtype=theano.config.floatX)
        
        delta_values = max_values - min_values
        for i in numpy.arange(ids_values.shape[0]):
            if mask[i]==0:
                ids_dx[i,] = 0
                ids_new_values[i,] = ids_values[i,]
            else:
                ids_new_values[i,] = ((ids_new_values[i,]-min_values)/delta_values - 0.5) * self.min_max_compresion_rate
                ids_dx[i,] = ids_new_values[i,] - ids_values[i,]
                
        #save_ids(ids_new_values,"items_new_ids.dta_001_new1.txt")
                
        avg_dx = numpy.zeros(shape=ids_values.shape,dtype=theano.config.floatX)
        avg_cnt = int(dx_cnt*avg_rate)
        lt = time.time()
        for i in numpy.arange(ids_values.shape[0]):
            if mask[i]==0:
                avg_dx[i,] = 0    
                continue
            dist = numpy.zeros(shape=(ids_values.shape[0]),dtype=theano.config.floatX)
            for j in numpy.arange(ids_values.shape[0]):
                if mask[j]==0:
                    dist[j] = self.max_distance
                    continue
                dist[j] = self.__distance(ids_values,i,j)
            sorted_indices = numpy.argsort(dist)
            
            # sum nearest
            sumw = 1.0e-8 
            for j in numpy.arange(avg_cnt):
                w = numpy.exp(-dist[sorted_indices[j]]/self.dist_sqrt_coef)
                sumw = sumw + w
                avg_dx[i,] = avg_dx[i,] + w*ids_dx[sorted_indices[j],]
            # add to sum dx with small weight
            w = item_weight
            sumw = sumw + w
            avg_dx[i,] = avg_dx[i,] + w*ids_dx[i,]
            
            avg_dx[i,] = avg_dx[i,] / sumw
            t1 = time.time()
            if t1>lt+1:
                sys.stdout.write("\t\t\t\t\t\t\t\t\t\r")
                sys.stdout.write("calc_%s_avg_dx %f %%\r" % (self.name,float(i)/float(ids_values.shape[0])*100))
                lt = lt+1
                
        for i in numpy.arange(ids_values.shape[0]):
            if mask[i]==0:
                ids_new_values[i,] = ids_values[i,]
                continue
            new_smooth_move = avg_dx[i,]
            new_rearrange = ids_dx[i,] - avg_dx[i,]
            ids_new_values[i,] = ids_values[i,] + (new_smooth_move*(1.0-p) + new_rearrange*p)
            mv = max(numpy.max(ids_new_values[i,]),abs(numpy.min(ids_new_values[i,])))
            if mv>0.5:
                ids_new_values[i,] = ids_new_values[i,] * 0.5/mv  
        
        #save_ids(ids_values,"items_new_ids.dta_001_old.txt")
        #save_ids(ids_new_values,"items_new_ids.dta_001_new.txt")
        #save_ids(avg_dx,"items_new_ids.dta_001_avg.txt")
             
        return ids_new_values
    
    def calc_new_ids(self,ids_values,ids_new_values,mask):
        return self.__calc_new_ids(
            ids_values = ids_values
            ,ids_new_values = ids_new_values
            ,avg_rate = self.avg_rate
            ,mask = mask
            ,p = self.normilized_vs_avg_rate
            ,item_weight =self.avg_dx_item_small_weight
            )

class MatrixIDs(object):
    '''
    The class to get IDs by the sparse matrix of the data (the both axes data and rates data in the matrix).  
    '''

    def __init__(self
        ,usersids_data,itemsids_data
        ,ratings_by_user,ratings_by_user_ids,ratings_by_user_idx
        ,ratings_by_item,ratings_by_item_ids,ratings_by_item_idx
        ,rng
        ,theano_rng
        ,consts = Consts()
        ,users_ids = None
        ,items_ids = None
        ):
        '''
        Constructor
        '''
        
        self.usersids_data = usersids_data
        self.itemsids_data = itemsids_data
        self.ratings_by_user = ratings_by_user
        self.ratings_by_user_ids = ratings_by_user_ids
        self.ratings_by_user_idx = ratings_by_user_idx
        self.ratings_by_item = ratings_by_item
        self.ratings_by_item_ids = ratings_by_item_ids
        self.ratings_by_item_idx = ratings_by_item_idx
        self.rng = rng
        
        self.user_indice = 0
        self.movie_indice = 1
        
        self.item_id_size = consts.item_id_size
        self.user_id_size = consts.user_id_size
        self.new_user_cycles = consts.new_user_cycles  
        self.new_item_cycles = consts.new_item_cycles  
        #self.users_ids_move_elem_count_rate = consts.users_ids_move_elem_count_rate
        #self.items_ids_move_elem_count_rate = consts.items_ids_move_elem_count_rate
        self.items_count = self.ratings_by_item_ids[self.ratings_by_item_idx[len(self.ratings_by_item_idx)-1,0],self.movie_indice]
        self.users_count = self.ratings_by_user_ids[self.ratings_by_user_idx[len(self.ratings_by_user_idx)-1,0],self.user_indice]
        if not items_ids:
            items_ids = rng.uniform(low = -0.5,high = 0.5, size = (self.items_count,self.item_id_size)).astype(theano.config.floatX)
        if not users_ids:
            users_ids = rng.uniform(low = -0.5,high = 0.5, size = (self.users_count,self.user_id_size)).astype(theano.config.floatX)  
        self.items_ids = items_ids
        self.items_ids_base = self.items_ids.copy()
        self.new_items_ids = self.items_ids.copy()
        self.users_ids = users_ids 
        self.users_ids_base = self.users_ids.copy()
        self.new_users_ids = self.users_ids.copy()
        
        
        self.mini_batch_size = consts.encoder_batch_size
        self.encoder_elements_count = consts.encode_elements_count 
        
        self.items_ids_to_user_id_autoencoder = AutoEncoder(
            mini_batch_size = self.mini_batch_size
            ,input_size = self.encoder_elements_count*(self.itemsids_data.shape[1]+self.item_id_size+ratings_by_user.shape[1]) 
            ,encoded_size = self.user_id_size
            ,hidden_count = consts.encoder_hidden_layers_count,hidden_size = consts.encoder_hidden_layer_size,activation = consts.encoder_hidden_layers_activation
            ,L1_decay = consts.encoder_L1_decay,L2_decay = consts.encoder_L2_decay
            ,numpy_rng = rng
            ,theano_rng = theano_rng
            ) #+usersids_data.shape[1]
        self.users_ids_to_item_id_autoencoder = AutoEncoder(
            mini_batch_size = self.mini_batch_size
            ,input_size = self.encoder_elements_count*(self.usersids_data.shape[1]+self.user_id_size+ratings_by_item.shape[1]) 
            ,encoded_size = self.item_id_size
            ,hidden_count = consts.encoder_hidden_layers_count,hidden_size = consts.encoder_hidden_layer_size,activation = consts.encoder_hidden_layers_activation
            ,L1_decay = consts.encoder_L1_decay,L2_decay = consts.encoder_L2_decay
            ,numpy_rng = rng
            ,theano_rng = theano_rng
            ) #+itemsids_data.shape[1]
        
        self.loss_users_to_item = float(0.0)
        self.loss_items_to_user = float(0.0)
        self.loss_k = consts.encoder_loss_k
        
        self.users_ids_updater = IDsUpdater(
            ids_count = self.users_count
            ,avg_rate = consts.users_ids_avg_rate
            ,normilized_vs_avg_rate = consts.ids_update_users_normilized_vs_avg_rate
            ,max_distance= consts.user_max_distance
            ,dist_sqrt_coef = consts.dist_sqrt_coef
            ,avg_dx_item_small_weight = consts.avg_dx_item_small_weight
            ,min_max_compresion_rate = consts.min_max_compresion_rate
            ,name = "users"
            )
        self.items_ids_updater = IDsUpdater(
            ids_count = self.items_count
            ,avg_rate = consts.items_ids_avg_rate
            ,normilized_vs_avg_rate = consts.ids_update_items_normilized_vs_avg_rate
            ,max_distance= consts.item_max_distance
            ,dist_sqrt_coef = consts.dist_sqrt_coef
            ,avg_dx_item_small_weight = consts.avg_dx_item_small_weight
            ,min_max_compresion_rate = consts.min_max_compresion_rate
            ,name = "items"
            ) 
        self.index = 0 
        #self.ids_move_count = consts.ids_move_count 
        #self.ids_move_count_coef = consts.ids_move_count_coef
        #self.ids_move_count = consts.ids_move_count 
        #self.ids_move_count_coef = consts.ids_move_count_coef
        
        self.itemsids_mini_batch_size = consts.itemids_batch_size 
        self.user_ids_rate_to_item_ids_net = ApproxNet(
            batch_size = self.itemsids_mini_batch_size
            ,input_size = self.usersids_data.shape[1]+self.user_id_size+ratings_by_user.shape[1]
            ,output_size = self.item_id_size
            ,hidden_count = consts.itemids_hidden_layers_count
            ,hidden_size = consts.itemids_hidden_layer_size
            ,hidden_activation = consts.itemids_hidden_layers_activation
            ,L1_decay = consts.itemids_L1_decay
            ,L2_decay = consts.itemids_L2_decay
            ,numpy_rng = rng
            ,theano_rng = theano_rng
            )
        self.itemids_loss_k = consts.itemids_loss_k
        self.loss_itemids = float(0.0) 
        
        return
    
    def update_user_ids(self,consts):
        i0 = numpy.int32(float(self.index)/float(consts.ids_move_count)*self.users_ids.shape[0])
        i1 = numpy.int32(float(self.index+1)/float(consts.ids_move_count)*self.users_ids.shape[0])
        if i0==i1:
            i1 = i0+1
        i0 = min(i0,self.users_ids.shape[0]-1)
        i1 = min(i1,self.users_ids.shape[0]-1)
        self.users_ids[i0:i1,] = self.new_users_ids[i0:i1,] 
        return
    
    def update_item_ids(self,consts):
        i0 = numpy.int32(float(self.index)/float(consts.ids_move_count)*self.items_ids.shape[0])
        i1 = numpy.int32(float(self.index+1)/float(consts.ids_move_count)*self.items_ids.shape[0])
        if i0==i1:
            i1 = i0+1
        i0 = min(i0,self.items_ids.shape[0]-1)
        i1 = min(i1,self.items_ids.shape[0]-1)
        self.items_ids[i0:i1,] = self.new_items_ids[i0:i1,]
        return
    
    def train_encoders(self,learning_rate,corruption_level,consts):
        encoder_size = self.encoder_elements_count*(self.itemsids_data.shape[1]+self.item_id_size+self.ratings_by_user.shape[1]) #+self.usersids_data.shape[1] 
        x_value = numpy.zeros((self.mini_batch_size,encoder_size), dtype=theano.config.floatX)
        for j in numpy.arange(self.mini_batch_size):
            user_idx1 = self.rng.randint(low=0 ,high = len(self.ratings_by_user_idx))
            curr_user_idx_index = curr_user_idx_index = self.ratings_by_user_idx[user_idx1,0]
            values = []
            user_id = self.ratings_by_item_ids[curr_user_idx_index,self.user_indice] - 1
            for k in numpy.arange(self.encoder_elements_count):
                item_idx1 = self.rng.randint(low = 0,high = self.ratings_by_user_idx[user_idx1,1]-curr_user_idx_index)
                rating_by_user_offs = curr_user_idx_index+item_idx1
                #user_id = self.ratings_by_user_ids[rating_by_user_offs,0] - 1
                item_id = self.ratings_by_user_ids[rating_by_user_offs,self.movie_indice] - 1
                values.append((self.items_ids[item_id,],self.itemsids_data[item_id,],self.ratings_by_user[rating_by_user_offs,]))
            values = sorted(values,key = lambda x : x[2][0])
            for k in numpy.arange(self.encoder_elements_count):
                i0 = (self.itemsids_data.shape[1]+self.item_id_size+self.ratings_by_user.shape[1])*k #
                i1 = i0 + self.item_id_size
                x_value[j,i0:i1] = values[k][0]
                i0 = i1;
                i1 = i1 + self.itemsids_data.shape[1]  
                x_value[j,i0:i1] = values[k][1]
                i0 = i1
                i1 = i1 + self.ratings_by_item.shape[1]
                x_value[j,i0:i1] = values[k][2][0]
            #i0 = (self.item_id_size+self.ratings_by_user.shape[1])*self.encoder_elements_count #self.itemsids_data.shape[1]+
            #i1 = i0 + self.usersids_data.shape[1]
            #x_value[j,i0:i1] = self.usersids_data[user_id,]

        loss = self.items_ids_to_user_id_autoencoder.train_fn(x_value,learning_rate,corruption_level)
        if loss[0]>=0:
            loss = numpy.sqrt(loss[0])
        else:
            loss = 0
        if self.loss_items_to_user==0:
            self.loss_items_to_user = loss
        else:
            self.loss_items_to_user += (loss - self.loss_items_to_user)*self.loss_k 
        
        encoder_size = self.encoder_elements_count*(self.usersids_data.shape[1]+self.user_id_size+self.ratings_by_item.shape[1]) #+self.itemsids_data.shape[1]
        x_value = numpy.zeros((self.mini_batch_size,encoder_size), dtype=theano.config.floatX)
        for j in numpy.arange(self.mini_batch_size):
            item_idx1 = self.rng.randint(low = 0,high = len(self.ratings_by_item_idx))
            curr_item_idx_index = self.ratings_by_item_idx[item_idx1,0]
            values = []
            item_id = self.ratings_by_item_ids[curr_item_idx_index,self.movie_indice] - 1   
            for k in numpy.arange(self.encoder_elements_count):
                user_idx1 = self.rng.randint(low = 0,high = self.ratings_by_item_idx[item_idx1,1]-curr_item_idx_index)
                rating_by_item_offs = curr_item_idx_index+user_idx1
                user_id = self.ratings_by_item_ids[rating_by_item_offs,self.user_indice] - 1
                #item_id = self.ratings_by_item_ids[rating_by_item_offs,1] - 1
                values.append((self.users_ids[user_id,],self.usersids_data[user_id,],self.ratings_by_item[rating_by_item_offs,]))
            values = sorted(values,key = lambda x : x[2][0])
            for k in numpy.arange(self.encoder_elements_count):
                i0 = (self.usersids_data.shape[1]+self.user_id_size+self.ratings_by_item.shape[1])*k # 
                i1 = i0 + self.user_id_size
                x_value[j,i0:i1] = values[k][0]
                i0 = i1;
                i1 = i1 + self.usersids_data.shape[1]  
                x_value[j,i0:i1] = values[k][1]
                i0 = i1
                i1 = i1 + self.ratings_by_user.shape[1]
                x_value[j,i0:i1] = values[k][2][0]
            #i0 = (self.user_id_size+self.ratings_by_user.shape[1])*self.encoder_elements_count #self.usersids_data.shape[1]+
            #i1 = i0 + self.itemsids_data.shape[1]
            #x_value[j,i0:i1] = self.itemsids_data[item_id,]
            
        loss = self.users_ids_to_item_id_autoencoder.train_fn(x_value,learning_rate,corruption_level)
        if loss[0]>=0:
            loss = numpy.sqrt(loss[0])
        else:
            loss = 0
        if self.loss_users_to_item==0:
            self.loss_users_to_item = loss
        else:
            self.loss_users_to_item += (loss - self.loss_users_to_item)*self.loss_k
            
            
        # update ids  
        self.update_user_ids(consts)
        self.update_item_ids(consts)
        
        self.index = self.index + 1
         
        return self.loss_items_to_user,self.loss_users_to_item
    
    def train_itemids(self,learning_rate,consts):
        input_size = self.usersids_data.shape[1]+self.user_id_size+self.ratings_by_user.shape[1]
        output_size = self.item_id_size
        x_value = numpy.zeros((self.itemsids_mini_batch_size,input_size),dtype=theano.config.floatX)
        y_target = numpy.zeros((self.itemsids_mini_batch_size,output_size),dtype=theano.config.floatX) 
        for bi in numpy.arange(self.itemsids_mini_batch_size):
            user_idx1 = self.rng.randint(low=0 ,high = len(self.ratings_by_user_idx))
            curr_user_idx_index = self.ratings_by_user_idx[user_idx1,0]
            rating_by_user_offs = self.rng.randint(low = curr_user_idx_index,high = self.ratings_by_user_idx[user_idx1,1])
            #user_id = self.ratings_by_user_ids[rating_by_user_offs,0] - 1
            user_id = self.ratings_by_item_ids[rating_by_user_offs,self.user_indice] - 1
            item_id = self.ratings_by_user_ids[rating_by_user_offs,self.movie_indice] - 1
            
            i0 = 0
            i1 = i0 + self.usersids_data.shape[1]
            x_value[bi,i0:i1] = self.usersids_data[user_id,:]
            i0 = i1
            i1 = i0 + self.user_id_size
            x_value[bi,i0:i1] = self.users_ids[user_id,:]
            i0 = i1
            i1 = i0 + self.ratings_by_user.shape[1]
            x_value[bi,i0:i1] = self.ratings_by_user[rating_by_user_offs,:]
            y_target[bi,:] = self.items_ids[item_id,:]
            pass
        
        loss = self.user_ids_rate_to_item_ids_net.train_fn(x_value,y_target,learning_rate)
        if loss[0]>=0:
            loss = numpy.sqrt(loss[0])
        else:
            loss = float(0)
        if self.loss_itemids==0:
            self.loss_itemids = loss
        else:
            self.loss_itemids += (loss - self.loss_itemids)*self.itemids_loss_k 
        return self.loss_itemids
    
    def __get_user_ids(self,user_idxes):
        encoder_size = self.encoder_elements_count*(self.itemsids_data.shape[1]+self.item_id_size+self.ratings_by_user.shape[1]) #+self.usersids_data.shape[1]  
        x_value = numpy.zeros((self.mini_batch_size,encoder_size), dtype=theano.config.floatX)
        for j in numpy.arange(self.mini_batch_size):
            user_idx1 = 0
            if j<len(user_idxes):
                user_idx1 = user_idxes[j]
            else:
                user_idx1 = user_idxes[-1]
            curr_user_idx_index = curr_user_idx_index = self.ratings_by_user_idx[user_idx1,0]
            values = []
            user_id = self.ratings_by_item_ids[curr_user_idx_index,self.user_indice] - 1
            for k in numpy.arange(self.encoder_elements_count):
                item_idx1 = self.rng.randint(low = 0,high = self.ratings_by_user_idx[user_idx1,1]-curr_user_idx_index)
                rating_by_user_offs = curr_user_idx_index+item_idx1
                #user_id = self.ratings_by_user_ids[rating_by_user_offs,0] - 1
                item_id = self.ratings_by_user_ids[rating_by_user_offs,self.movie_indice] - 1
                values.append((self.items_ids[item_id,],self.itemsids_data[item_id,],self.ratings_by_user[rating_by_user_offs,]))
            values = sorted(values,key = lambda x : x[2][0])
            for k in numpy.arange(self.encoder_elements_count):
                i0 = (self.itemsids_data.shape[1]+self.item_id_size+self.ratings_by_user.shape[1])*k #  
                i1 = i0 + self.item_id_size
                x_value[j,i0:i1] = values[k][0]
                i0 = i1;
                i1 = i1 + self.itemsids_data.shape[1]  
                x_value[j,i0:i1] = values[k][1]
                i0 = i1
                i1 = i1 + self.ratings_by_item.shape[1]
                x_value[j,i0:i1] = values[k][2][0]
            #i0 = (self.item_id_size+self.ratings_by_user.shape[1])*self.encoder_elements_count # self.itemsids_data.shape[1]+
            #i1 = i0 + self.usersids_data.shape[1]
            #x_value[j,i0:i1] = self.usersids_data[user_id,]

        y_result = self.items_ids_to_user_id_autoencoder.get_encoded_fn(x_value)
        return y_result[0]
    
    def __get_item_ids(self,item_idxes):
        encoder_size = self.encoder_elements_count*(self.usersids_data.shape[1]+self.user_id_size+self.ratings_by_item.shape[1]) #+self.itemsids_data.shape[1] 
        x_value = numpy.zeros((self.mini_batch_size,encoder_size), dtype=theano.config.floatX)
        for j in numpy.arange(self.mini_batch_size):
            item_idx1 = 0
            if j<len(item_idxes):
                item_idx1 = item_idxes[j]
            else:
                item_idx1 = item_idxes[-1]
            curr_item_idx_index = self.ratings_by_item_idx[item_idx1,0]
            values = []
            item_id = self.ratings_by_item_ids[curr_item_idx_index,self.movie_indice] - 1   
            for k in numpy.arange(self.encoder_elements_count):
                user_idx1 = self.rng.randint(low = 0,high = self.ratings_by_item_idx[item_idx1,1]-curr_item_idx_index)
                rating_by_item_offs = curr_item_idx_index+user_idx1
                user_id = self.ratings_by_item_ids[rating_by_item_offs,self.user_indice] - 1
                #item_id = self.ratings_by_item_ids[rating_by_item_offs,1] - 1
                values.append((self.users_ids[user_id,],self.usersids_data[user_id,],self.ratings_by_item[rating_by_item_offs,]))
            values = sorted(values,key = lambda x : x[2][0])
            for k in numpy.arange(self.encoder_elements_count):
                i0 = (self.usersids_data.shape[1]+self.user_id_size+self.ratings_by_item.shape[1])*k #  , 
                i1 = i0 + self.user_id_size
                x_value[j,i0:i1] = values[k][0]
                i0 = i1;
                i1 = i1 + self.usersids_data.shape[1]  
                x_value[j,i0:i1] = values[k][1]
                i0 = i1
                i1 = i1 + self.ratings_by_user.shape[1]
                x_value[j,i0:i1] = values[k][2][0]
            #i0 = (self.user_id_size+self.ratings_by_user.shape[1])*self.encoder_elements_count #self.usersids_data.shape[1]+
            #i1 = i0 + self.itemsids_data.shape[1]
            #x_value[j,i0:i1] = self.itemsids_data[item_id,]
            
        y_result = self.users_ids_to_item_id_autoencoder.get_encoded_fn(x_value)
        return y_result[0]
    
    def get_new_user_ids(self,consts):
        lt = time.time()
        self.users_ids_base = self.users_ids.copy()
        self.new_users_ids = self.users_ids.copy()
        user_ids_ind = numpy.arange(len(self.ratings_by_user_idx))
        numpy.random.shuffle(user_ids_ind)
        updates_elements_cnt = numpy.int32(len(user_ids_ind)*consts.users_ids_move_elem_count_rate)
        updates_elements_cnt = ((updates_elements_cnt/self.mini_batch_size)+1)*self.mini_batch_size
        for cycle in numpy.arange(self.new_user_cycles):
            for i00 in numpy.arange(updates_elements_cnt/self.mini_batch_size):
                i0 = i00*self.mini_batch_size
                i1 = (i00+1)*self.mini_batch_size
                if i1>=updates_elements_cnt:
                    i1 = updates_elements_cnt-1
                #user_ids = self.ratings_by_user_ids[self.ratings_by_user_idx[user_ids_ind[i0:i1],0],0]
                encoded = self.__get_user_ids(user_ids_ind[i0:i1])
                for i in numpy.arange(self.mini_batch_size):
                    if i0+i>=updates_elements_cnt:
                        continue
                    user_id = self.ratings_by_user_ids[self.ratings_by_user_idx[user_ids_ind[i0+i],0],self.user_indice] - 1
                    if cycle==0:
                        self.new_users_ids[user_id,] = encoded[i,] 
                    else:
                        self.new_users_ids[user_id,] = self.new_users_ids[user_id,] + encoded[i,]
                t1 = time.time()
                if t1>lt+1:
                    rate = float(i00+cycle*updates_elements_cnt/self.mini_batch_size)/float(updates_elements_cnt/self.mini_batch_size*self.new_user_cycles)
                    sys.stdout.write("\t\t\t\t\t\t\t\t\t\r")
                    sys.stdout.write("get_new_user_ids %f %%\r" % (rate*100))
                    lt = lt+1
                    
        user_ids = self.ratings_by_user_ids[self.ratings_by_user_idx[user_ids_ind[0:updates_elements_cnt],0],self.user_indice] - 1
                        
        user_id_max = self.ratings_by_user_ids[self.ratings_by_user_idx[len(self.ratings_by_user_idx)-1,0],self.user_indice]
        self.users_ids_mask = numpy.zeros(shape=user_id_max, dtype=numpy.int8)
        self.users_ids_mask[user_ids] = 1
        self.new_users_ids[user_ids,] = self.new_users_ids[user_ids,] / float(self.new_user_cycles)
        
        #for i in numpy.arange(updates_elements_cnt):
        #    user_id = user_ids_ind[i]
        #    self.new_users_ids[user_id,] = self.new_users_ids[user_id,] / float(self.new_user_cycles)
            
        self.new_users_ids = self.users_ids_updater.calc_new_ids(
            ids_values = self.users_ids_base
            ,ids_new_values = self.new_users_ids
            ,mask = self.users_ids_mask
            )
        
        updates_elements_cnt_lo = numpy.int32(updates_elements_cnt*consts.users_ids_move_elem_count_rate1)
        user_ids = self.ratings_by_user_ids[self.ratings_by_user_idx[user_ids_ind[updates_elements_cnt_lo:updates_elements_cnt],0],self.user_indice] - 1
        self.new_users_ids[user_ids,] = self.users_ids_base[user_ids,]
         
        p=consts.ids_move_count_coef
        self.new_users_ids = (1.0 - p)*self.users_ids_base + p*self.new_users_ids
        self.index = 0
        return

    def get_new_item_ids(self,consts):
        lt = time.time()
        self.items_ids_base =  self.items_ids.copy()
        self.new_items_ids = self.items_ids.copy()
        item_ids_ind = numpy.arange(len(self.ratings_by_item_idx))
        numpy.random.shuffle(item_ids_ind)
        updates_elements_cnt = numpy.int32(len(item_ids_ind)*consts.items_ids_move_elem_count_rate)
        updates_elements_cnt = ((updates_elements_cnt/self.mini_batch_size)+1)*self.mini_batch_size 
        for cycle in numpy.arange(self.new_item_cycles):
            for i00 in numpy.arange(updates_elements_cnt/self.mini_batch_size):
                i0 = i00*self.mini_batch_size
                i1 = (i00+1)*self.mini_batch_size
                if i1>=updates_elements_cnt:
                    i1 = updates_elements_cnt-1
                #item_ids = self.ratings_by_item_ids[self.ratings_by_item_idx[item_ids_ind[i0:i1],0],1]
                encoded = self.__get_item_ids(item_ids_ind[i0:i1])
                for i in numpy.arange(self.mini_batch_size):
                    if i0+i>=updates_elements_cnt:
                        continue
                    item_id = self.ratings_by_item_ids[self.ratings_by_item_idx[item_ids_ind[i0+i],0],self.movie_indice] - 1 
                    if cycle==0:
                        self.new_items_ids[item_id,] = encoded[i,] 
                    else:
                        self.new_items_ids[item_id,] = self.new_items_ids[item_id,] + encoded[i,]
                t1 = time.time()
                if t1>lt+1:
                    rate = float(i00+cycle*updates_elements_cnt/self.mini_batch_size)/float(updates_elements_cnt/self.mini_batch_size*self.new_item_cycles)
                    sys.stdout.write("\t\t\t\t\t\t\t\t\t\r")
                    sys.stdout.write("get_new_item_ids %f %%\r" % (rate*100))
                    lt = lt+1
             
        item_ids = self.ratings_by_item_ids[self.ratings_by_item_idx[item_ids_ind[0:updates_elements_cnt],0],self.movie_indice] - 1
                   
        item_id_max = self.ratings_by_item_ids[self.ratings_by_item_idx[len(self.ratings_by_item_idx)-1,0],self.movie_indice]
        self.items_ids_mask = numpy.zeros(shape=item_id_max, dtype=numpy.int8)
        self.items_ids_mask[item_ids] = 1
        self.new_items_ids[item_ids,] = self.new_items_ids[item_ids,] / float(self.new_item_cycles)
        
        self.new_items_ids = self.items_ids_updater.calc_new_ids(
            ids_values = self.items_ids_base
            ,ids_new_values = self.new_items_ids
            ,mask = self.items_ids_mask
            )

        updates_elements_cnt_lo = numpy.int32(updates_elements_cnt*consts.items_ids_move_elem_count_rate1)
        item_ids = self.ratings_by_item_ids[self.ratings_by_item_idx[item_ids_ind[updates_elements_cnt_lo:updates_elements_cnt],0],self.movie_indice] - 1
        self.new_items_ids[item_ids,] = self.items_ids_base[item_ids,]
        
        p=consts.ids_move_count_coef
        self.new_items_ids = (1.0 - p)*self.items_ids_base + p*self.new_items_ids
        self.index = 0
        return
    
    def save(self,index,consts):
        numpy.save(file = consts.get_file_name_by_index(index,consts.users_ids_file_name), arr = self.users_ids)
        numpy.save(file = consts.get_file_name_by_index(index,consts.items_ids_file_name), arr = self.items_ids)
        self.users_ids_to_item_id_autoencoder.save_state(
            file_name = consts.get_file_name_by_index(index,consts.users_ids_to_item_id_autoencoder_file_name), 
            consts = consts)
        self.items_ids_to_user_id_autoencoder.save_state(
            file_name = consts.get_file_name_by_index(index,consts.items_ids_to_user_id_autoencoder_file_name)
            ,consts = consts
            )
        self.user_ids_rate_to_item_ids_net.save_state(
            file_name = consts.get_file_name_by_index(index,consts.user_ids_rate_to_item_ids_net_file_name)
            ,consts = consts
            )
        return
    
    def load(self,index,consts):
        if not self.users_ids_to_item_id_autoencoder.load_state(
                file_name=consts.get_file_name_by_index(index,consts.users_ids_to_item_id_autoencoder_file_name)
                ,consts=consts):
            return False
        if not self.items_ids_to_user_id_autoencoder.load_state(
                file_name=consts.get_file_name_by_index(index,consts.items_ids_to_user_id_autoencoder_file_name)
                , consts=consts):
            return False
        if not self.user_ids_rate_to_item_ids_net.load_state(
                file_name=consts.get_file_name_by_index(index,consts.user_ids_rate_to_item_ids_net_file_name)
                ,consts = consts):
            return False
        file_name = consts.get_file_name_by_index(index,consts.users_ids_file_name)
        if not os.path.isfile(path = file_name):
            return False
        data = numpy.load(file = file_name)
        self.users_ids = numpy.asarray(a = data,dtype=theano.config.floatX)
        file_name = consts.get_file_name_by_index(index,consts.items_ids_file_name)
        if not os.path.isfile(path = file_name):
            return False
        data = numpy.load(file = file_name)
        self.items_ids = numpy.asarray(a = data,dtype=theano.config.floatX)
        self.items_ids_base = self.items_ids.copy()
        self.new_items_ids = self.items_ids.copy()
        self.users_ids_base = self.users_ids.copy()
        self.new_users_ids = self.users_ids.copy()
        return True
 
class RatesApprox(object):
    ''' class for rates approximation'''
    
    def __init__(self
        ,usersids_data,itemsids_data
        ,ratings_by_user,ratings_by_user_ids,ratings_by_user_idx
        ,rng
        ,theano_rng
        ,matrix_ids
        ,consts = Consts()
        ):
        
        self.mini_batch_size = consts.result_batch_size
        self.usersids_data = usersids_data
        self.itemsids_data = itemsids_data
        self.ratings_by_user = ratings_by_user
        self.ratings_by_user_ids = ratings_by_user_ids
        self.ratings_by_user_idx = ratings_by_user_idx
        self.rng = rng

        self.user_indice = 0
        self.movie_indice = 1
        
        self.item_id_size = consts.item_id_size
        self.user_id_size = consts.user_id_size
        self.users_count = self.ratings_by_user_ids[self.ratings_by_user_idx[len(self.ratings_by_user_idx)-1,0],self.user_indice]
        self.matrix_ids = matrix_ids
        
        self.net = ApproxNet(
            batch_size = self.mini_batch_size
            ,input_size = self.item_id_size + self.user_id_size + self.itemsids_data.shape[1]+self.usersids_data.shape[1]+self.ratings_by_user.shape[1] - 1   
            ,output_size = 1
            ,hidden_count = consts.result_hidden_layers_count
            ,hidden_size = consts.result_hidden_layer_size
            ,hidden_activation = consts.result_hidden_layers_activation
            ,numpy_rng = rng 
            ,theano_rng = theano_rng
            ,L1_decay = consts.result_L1_decay
            ,L2_decay = consts.result_L2_decay
            )
        
        self.all_rates_indices = numpy.arange(self.ratings_by_user_ids.shape[0])
        numpy.random.shuffle(self.all_rates_indices)
        self.train_size_rate = consts.train_rate
        self.loss = float(0.0)
        self.loss_k = consts.result_loss_k
        return

    def train(self,learning_rate):
        x_size = self.item_id_size + self.user_id_size + self.itemsids_data.shape[1]+self.usersids_data.shape[1]+self.ratings_by_user.shape[1] - 1 
        x_value = numpy.zeros((self.mini_batch_size,x_size), dtype=theano.config.floatX)
        y_size = 1
        y_value = numpy.zeros((self.mini_batch_size,y_size), dtype=theano.config.floatX)
        max_train_indice = numpy.int32(self.all_rates_indices.shape[0]*self.train_size_rate)
        for bi in numpy.arange(self.mini_batch_size):
            train_indice = self.rng.randint(low=0 ,high = max_train_indice) 
            user_id = self.ratings_by_user_ids[self.all_rates_indices[train_indice],self.user_indice] - 1 
            item_id = self.ratings_by_user_ids[self.all_rates_indices[train_indice],self.movie_indice] - 1 
            i0 = 0
            i1 = self.item_id_size
            x_value[bi,i0:i1] = self.matrix_ids.items_ids[item_id,]
            i0 = i1 
            i1 = i0 + self.user_id_size
            x_value[bi,i0:i1] = self.matrix_ids.users_ids[user_id,]
            i0 = i1
            i1 = i0 + self.itemsids_data.shape[1]
            x_value[bi,i0:i1] = self.itemsids_data[item_id,]
            i0 = i1
            i1 = i0 + self.usersids_data.shape[1]
            x_value[bi,i0:i1] = self.usersids_data[user_id,]
            i0 = i1
            i1 = i0 + self.ratings_by_user.shape[1] - 1
            x_value[bi,i0:i1] = self.ratings_by_user[self.all_rates_indices[train_indice],1:]
            y_value[bi,0] = self.ratings_by_user[self.all_rates_indices[train_indice],0] 
                
        loss = self.net.train_fn(x_value,y_value,learning_rate)
        if loss[0]>=0:
            loss = numpy.sqrt(loss[0])
        else:
            loss = 0
        if self.loss==0:
            self.loss = loss
        else:
            self.loss += (loss - self.loss)*self.loss_k  
        return self.loss
    
    def get_rates(self,userids_itemids,ratesinfo):
        rates = numpy.zeros(userids_itemids.shape[0],dtype=theano.config.floatX)
        x_size = self.item_id_size + self.user_id_size + self.itemsids_data.shape[1]+self.usersids_data.shape[1]+self.ratings_by_user.shape[1] - 1 
        x_value = numpy.zeros((self.mini_batch_size,x_size), dtype=theano.config.floatX)
        cnt = (userids_itemids.shape[0]/self.mini_batch_size+1)*self.mini_batch_size
        for i in numpy.arange(cnt/self.mini_batch_size):
            for bi in numpy.arange(self.mini_batch_size):
                idx = i*self.mini_batch_size + bi
                if idx>=userids_itemids.shape[0]:
                    idx = userids_itemids.shape[0]-1
                user_id = userids_itemids[idx,self.user_indice]
                item_id = userids_itemids[idx,self.movie_indice]
                i0 = 0
                i1 = self.item_id_size
                x_value[bi,i0:i1] = self.matrix_ids.items_ids[item_id,]
                i0 = i1 
                i1 = i0 + self.user_id_size
                x_value[bi,i0:i1] = self.matrix_ids.users_ids[user_id,]
                i0 = i1
                i1 = i0 + self.itemsids_data.shape[1]
                x_value[bi,i0:i1] = self.itemsids_data[item_id,]
                i0 = i1
                i1 = i0 + self.usersids_data.shape[1]
                x_value[bi,i0:i1] = self.usersids_data[user_id,]
                i0 = i1
                i1 = i0 + self.ratings_by_user.shape[1] - 1
                x_value[bi,i0:i1] = ratesinfo[idx,:]
            y_result = self.net.run_fn(x_value)
            for bi in numpy.arange(self.mini_batch_size):
                idx = i*self.mini_batch_size + bi
                if idx>=userids_itemids.shape[0]:
                    continue
                rates[idx] = y_result[0][bi,0]
        return rates
    
    def validate(self,consts):
        max_train_indice = numpy.int32(self.all_rates_indices.shape[0]*self.train_size_rate)
        validate_indicies = numpy.arange(self.all_rates_indices.shape[0] - max_train_indice)
        validate_indicies[:] += max_train_indice
        userids_itemids = numpy.zeros((self.all_rates_indices.shape[0] - max_train_indice,2),dtype=theano.config.floatX);
        rates_info = numpy.zeros((self.all_rates_indices.shape[0] - max_train_indice,self.ratings_by_user.shape[1] - 1),dtype=theano.config.floatX)
        i0 = 0
        for validate_indice in validate_indicies:
            userids_itemids[i0,self.user_indice] = self.ratings_by_user_ids[self.all_rates_indices[validate_indice],self.user_indice] - 1
            userids_itemids[i0,self.movie_indice] = self.ratings_by_user_ids[self.all_rates_indices[validate_indice],self.movie_indice] - 1
            rates_info[i0,:] = self.ratings_by_user[self.all_rates_indices[validate_indice],1:]  
            i0 += 1
            pass
        rates = self.get_rates(userids_itemids,rates_info)
        loss = 0
        i0 = 0
        for validate_indice in validate_indicies:
            loss += numpy.square(self.ratings_by_user[self.all_rates_indices[validate_indice],0] - rates[i0])
            i0 += 1
            pass
        return numpy.sqrt(loss/rates.shape[0])
    
    def save(self,index,consts):
        self.net.save_state(file_name = consts.get_file_name_by_index(index,consts.result_net_file_name), consts = consts)
        return
    
    def load(self,index,consts):
        self.net.load_state(file_name = consts.get_file_name_by_index(index,consts.result_net_file_name), consts = consts)
        return
    
    pass
    
 
def get_aranged(value,min_value,max_value):
    if abs(max_value-min_value)<1e-9: 
        return 0
    return (float(value)-float(min_value))/(float(max_value)-float(min_value)) - float(0.5)        
        
class RecommenderSystem(object):
    '''
    class for recommender systems
    '''
    
    def prepare_data(self,consts = Consts()):
        print("loading data...")
        
        self.user_indice = 0
        self.movie_indice = 1
        
        
        # user_cvs
        # columns: 
        #     id -- int (key); sex -- ['M'|'F']; age -- int; 
        #     accupation -- int; lattitude -- real; longitude -- real; 
        #     timezone -- int; dts -- [0|1];   
        #
        users_cvs = pandas.read_csv(
            consts.users_cvs_file_name
            ,header=None
            ,sep=";"
            ,names = ["id","sex","age","accupation","lattitude","longitude","timezone","dts"]
            ,skipinitialspace = False
            )
        print("The users_cvs was loaded.")
        #print(users_cvs)
        
        # movies_cvs 
        # columns:
        #    id -- int (key); name -- string; gender -- string; year -- int;
        movies_cvs = pandas.read_csv(
            consts.movies_cvs_file_name
            ,header=None
            ,sep=";"
            ,names = ["id","name","gender","year"]
            ,skipinitialspace = False
            )
        print("The movies_cvs was loaded.")
        #print(movies_cvs)
        
        # ratings_cvs
        # columns:
        #     userid -- int (from users_cvs id key); filmid -- int (from movies_cvs id key); 
        #     rate -- real; wday -- int; yday -- int; year -- int; 
        ratings_cvs = pandas.read_csv(
            consts.ratings_cvs_file_name
            ,header=None
            ,sep=";"
            ,names=["userid","filmid","rate","wday","yday","year"]
            ,skipinitialspace = False
            )
        print("The ratings_cvs was loaded.")
        #print(ratings_cvs)
        
        
        # usersids
        # columns:
        #     sex -- +0.5 - 'M', -0.5 - 'F'
        #     age -- -0.5 - min, +0.5 - max
        
        last_user_id = users_cvs["id"][len(users_cvs)-1]
        usersids = numpy.zeros(dtype=theano.config.floatX,shape=(last_user_id,2))
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
        moviesids = numpy.zeros(dtype=theano.config.floatX,shape=(last_film_id,1))
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
           
        ratings_by_user = numpy.zeros(dtype=theano.config.floatX,shape=(len(ratings_cvs),2))
        ratings_by_user_ids = numpy.zeros(dtype=numpy.int32,shape=(len(ratings_cvs),2))
        ratings_by_user_idx = numpy.zeros(dtype=numpy.int32,shape=(len(grouped_by_user),2))
        i = 0
        li = 0
        lt = time.time()
        j = 0
        for name,group in grouped_by_user:
            user_id = numpy.int32(name)
            for row_id in group["id"]:
                ratings_by_user_ids[j,self.user_indice] = user_id 
                ratings_by_user_ids[j,self.movie_indice] = numpy.int32(ratings_cvs.loc[row_id,"filmid"])
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
        ratings_by_movie = numpy.zeros(dtype=theano.config.floatX,shape=(len(ratings_cvs),2))
        ratings_by_movie_ids = numpy.zeros(dtype=numpy.int32,shape=(len(ratings_cvs),2))
        ratings_by_movie_idx = numpy.zeros(dtype=numpy.int32,shape=(len(group_by_movie),2))
        i = 0
        li = 0
        lt = time.time()
        j = 0
        for name,group in group_by_movie:
            film_id = numpy.int32(name)
            for row_id in group["id"]:
                ratings_by_movie_ids[j,self.user_indice] = numpy.int32(ratings_cvs.loc[row_id,"userid"])
                ratings_by_movie_ids[j,self.movie_indice] = film_id 
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
    
    def load_data(self,consts=Consts()):
        self.usersids = numpy.load(file=consts.userids_npy_file_name)
        self.moviesids = numpy.load(file=consts.moviesids_npy_file_name)
        self.ratings_by_user = numpy.load(file=consts.ratings_by_user_npy_file_name)
        self.ratings_by_user_ids = numpy.load(file=consts.ratings_by_user_ids_npy_file_name)
        self.ratings_by_user_idx = numpy.load(file=consts.ratings_by_user_idx_npy_file_name)
        self.ratings_by_movie = numpy.load(file=consts.ratings_by_movie_npy_file_name)
        self.ratings_by_movie_ids = numpy.load(file=consts.ratings_by_movie_ids_npy_file_name)
        self.ratings_by_movie_idx = numpy.load(file=consts.ratings_by_movie_idx_npy_file_name)
        return
     
    def __init__(self
        ,rng
        ,theano_rng
        ,consts = Consts()
        ):
        '''
        The constructor
        '''
        
        self.load_data()
        self.matrix_ids = MatrixIDs(
            usersids_data = self.usersids,itemsids_data = self.moviesids
            ,ratings_by_user = self.ratings_by_user,ratings_by_user_ids = self.ratings_by_user_ids,ratings_by_user_idx = self.ratings_by_user_idx
            ,ratings_by_item = self.ratings_by_movie,ratings_by_item_ids = self.ratings_by_movie_ids,ratings_by_item_idx = self.ratings_by_movie_idx
            ,rng = rng
            ,theano_rng = theano_rng
            ,consts = consts
            ,users_ids = None
            ,items_ids = None
            )
        
        self.rates_approx = RatesApprox(
            usersids_data = self.usersids,itemsids_data = self.moviesids
            ,ratings_by_user = self.ratings_by_user,ratings_by_user_ids = self.ratings_by_user_ids,ratings_by_user_idx = self.ratings_by_user_idx
            ,rng = rng
            ,theano_rng = theano_rng
            ,matrix_ids = self.matrix_ids
            ,consts = consts
            )
        
        if consts.load_from_ids>0:
            self.matrix_ids.load(index = consts.load_from_ids, consts = consts)
            self.rates_approx.load(index = consts.load_from_ids, consts = consts)
        return
    
    def train_encoders(self,learning_rate,corruption_level,consts): #,print_flag,idx
        loss_items_to_user,loss_users_to_item = self.matrix_ids.train_encoders(learning_rate = learning_rate, corruption_level = corruption_level,consts = consts)
        return loss_items_to_user,loss_users_to_item
    
    def train_itemids(self,learning_rate,consts):
        loss = self.matrix_ids.train_itemids(learning_rate = learning_rate,consts = consts)
        return loss
    
    def train_rates(self,learning_rate):
        loss = self.rates_approx.train(learning_rate)
        return loss
    
    def validate_rates(self,consts):
        return self.rates_approx.validate(consts)
        
    def calc_new_ids(self,consts):
        self.matrix_ids.get_new_user_ids(consts)
        self.matrix_ids.get_new_item_ids(consts)
        return
    
    def save(self,index,consts):
        self.matrix_ids.save(index,consts)
        return
    
    def save_rates(self,index,consts):
        self.rates_approx.save(index,consts)
        return
    
    def load(self,index,consts):
        self.matrix_ids.load(index = consts.load_from_ids, consts = consts)
        return
    
    pass # class RecommenderSystem
  
class NearestMovies(object):
    def __init__(self,ids_index = 0,consts = Consts()):
        self.movies_cvs = pandas.read_csv(
            consts.movies_cvs_file_name
            ,header=None
            ,sep=";"
            ,names = ["id","name","gender","year"]
            ,skipinitialspace = False
            )
        items_ids1 = numpy.load(file = consts.get_file_name_by_index(ids_index,consts.items_ids_file_name))
        self.items_ids = numpy.zeros_like(a = items_ids1)
        self.items_ids[self.movies_cvs["id"]-1,] = items_ids1[self.movies_cvs["id"]-1,]
        self.users_ids = numpy.load(file = consts.get_file_name_by_index(ids_index,consts.users_ids_file_name))
        self.max_distance = consts.item_max_distance 
        #print("")
        #print()
        return
    
    def save_dta(self,index,consts):
        filename = consts.get_file_name_by_index(index,consts.users_ids_dta_file_name)
        outfile = open(filename,"wt")
        for i in numpy.arange(self.users_ids.shape[0]):
            outfile.write("%d\t" % (i,))
            for j in numpy.arange(self.users_ids.shape[1]):
                outfile.write("%f\t" % (self.users_ids[i,j],))
            outfile.write("\n")
            
        filename = consts.get_file_name_by_index(index,consts.items_ids_dta_file_name)
        outfile = open(filename,"wt")
        for i in numpy.arange(self.items_ids.shape[0]):
            outfile.write("%d\t" % (i,))
            for j in numpy.arange(self.items_ids.shape[1]):
                outfile.write("%f\t" % (self.items_ids[i,j],))
            outfile.write("\n")
        return
    
    def __distance(self,ids_values,i0,i1):
        df1 = self.movies_cvs[self.movies_cvs.id==i0]
        df2 = self.movies_cvs[self.movies_cvs.id==i1]
        if df1.empty or df2.empty:
            return self.max_distance
        return numpy.sum(numpy.square(ids_values[i0,]-ids_values[i1,]))
    
    def get_nearest_movies(self,movie_id):
        dist = numpy.zeros(shape=self.items_ids.shape[0],dtype=theano.config.floatX)
        for i in numpy.arange(self.items_ids.shape[0]):
            dist[i] = self.__distance(ids_values = self.items_ids, i0 = movie_id-1, i1 = i)
        indices = numpy.argsort(a = dist)
        return [
                (dist[indice]
                 ,indice+1
                 ,self.movies_cvs[self.movies_cvs["id"]==(indice+1)]["name"].values[0]
                 ,self.movies_cvs[self.movies_cvs["id"]==(indice+1)]["gender"].values[0]
                 ) for indice in indices[0:min(len(indices),200)] if not self.movies_cvs[self.movies_cvs["id"]==(indice+1)].empty]
        
    def __distance_to_center(self,centers,ci,i0):
        return numpy.sum(numpy.square(centers[ci,]-self.items_ids[i0,]))
    
    def __get_indexes(self,cluster_number,centers):
            indexes = [[] for i in numpy.arange(cluster_number)]
            for i in numpy.arange(self.items_ids.shape[0]):
                min_dist = 0
                cimin = 0
                for ci in numpy.arange(centers.shape[0]):
                    dist = self.__distance_to_center(centers,ci,i)
                    if ci==0:
                        min_dist = dist
                        cimin = 0
                    else:
                        if dist<min_dist:
                            min_dist = dist
                            cimin = ci 
                indexes[cimin].append(i)
            return indexes
        
    def get_clusters(self,cluster_number,iterations = 1000):
        lt = time.time()
        rng = numpy.random.RandomState()
        # initialize centers
        indices = numpy.asarray(a = (self.movies_cvs["id"]-1).values)
        rng.shuffle(indices)
        centers = self.items_ids[indices[0:cluster_number]]
        for i1 in numpy.arange(iterations):
            indexes = self.__get_indexes(cluster_number,centers)
            for ci in numpy.arange(centers.shape[0]):
                centers[ci,] = numpy.sum(self.items_ids[indexes[ci],],axis = 0)/float(len(indexes[ci]))
            t1 = time.time()
            if t1>lt+1:
                sys.stdout.write("\t\t\t\t\t\t\t\t\t\r")
                sys.stdout.write("get_clusters %f %%\r" % (float(i1)/float(iterations)*100))
                lt = lt+1
        indexes = self.__get_indexes(cluster_number,centers)
        res = []
        for ind in indexes:
            res.append(
                [
                 (indice+1
                    ,self.movies_cvs[self.movies_cvs["id"]==(indice+1)]["name"].values[0]
                    ,self.movies_cvs[self.movies_cvs["id"]==(indice+1)]["gender"].values[0]
                    ) 
                 for indice in ind if not self.movies_cvs[self.movies_cvs["id"]==(indice+1)].empty]
                )
        return res
    pass #class NearestMovies
     
class UserLines(object):
    '''
    the class of user lines of rates
    '''
    
    def __init__(self,rng,theano_rng,consts):
        '''
        constructor
        '''
        self.recommender_system = RecommenderSystem(rng= rng,theano_rng = theano_rng,consts = consts)
        self.matrix_ids = self.recommender_system.matrix_ids
        self.rates_approx = self.recommender_system.rates_approx
        self.rng = rng
        
        self.movies_cvs = pandas.read_csv(
            consts.movies_cvs_file_name
            ,header=None
            ,sep=";"
            ,names = ["id","name","gender","year"]
            ,skipinitialspace = False
            )
        return
    
    def __user_dist(self,user_id1,user_id2,dist):
        dist[user_id2] = numpy.sum(numpy.square(self.matrix_ids.users_ids[user_id1,:] - self.matrix_ids.users_ids[user_id2,:])) 
        return
    
    def __item_dist(self,item_id,item_ids,dist):
        dist[item_id] = numpy.sum(numpy.square(self.matrix_ids.items_ids[item_id] - item_ids))
        return 
    
    def __find_nearest(self,user_id,users_cnt):
        dist = numpy.zeros((self.matrix_ids.users_count,),dtype=theano.config.floatX)
        
        for user_id2 in numpy.arange(self.matrix_ids.users_count):
            self.__user_dist(user_id,user_id2,dist)
            pass
        if users_cnt>=self.matrix_ids.users_count:
            users_cnt = self.matrix_ids.users_count 
        sorted_indicies = numpy.argsort(dist)
        user_ids = [user_id1 for user_id1 in sorted_indicies[0:users_cnt]]
        #print(dist[sorted_indicies[0:users_cnt]])
        return user_ids
    
    def __calc_line(self,user_id,rating_info):
        line = numpy.zeros((self.matrix_ids.itemsids_mini_batch_size,self.matrix_ids.item_id_size),dtype=theano.config.floatX)
        
        input_size = self.matrix_ids.usersids_data.shape[1]+self.matrix_ids.user_id_size+self.matrix_ids.ratings_by_user.shape[1]
        input_x = numpy.zeros((self.matrix_ids.itemsids_mini_batch_size,input_size),dtype=theano.config.floatX) 
       
        for bi in numpy.arange(self.matrix_ids.itemsids_mini_batch_size):
            i0 = 0
            i1 = i0 + self.matrix_ids.usersids_data.shape[1]
            input_x[bi,i0:i1] = self.matrix_ids.usersids_data[user_id,:]
            i0 = i1
            i1 = i0 + self.matrix_ids.user_id_size
            input_x[bi,i0:i1] = self.matrix_ids.users_ids[user_id,:]
            i0 = i1
            i1 = i0 + self.matrix_ids.ratings_by_item.shape[1]
            input_x[bi,i0+1:i1] = rating_info[:]
            input_x[bi,i0] = (float(bi)/float(self.matrix_ids.itemsids_mini_batch_size) - float(0.5))/float(2.0)
            pass
        
        result_y = self.matrix_ids.user_ids_rate_to_item_ids_net.run_fn(input_x)
        print(input_x[0,:])
        print(result_y[0][0,:])
        for bi in numpy.arange(self.matrix_ids.itemsids_mini_batch_size):
            line[bi,:] = result_y[0][bi,:] 
            pass
        
        return line
    
    def __get_line_items(self,line):
        item_ids = numpy.zeros(line.shape[0],dtype="int32")
        dist = numpy.zeros((self.matrix_ids.items_count,),dtype=theano.config.floatX)
        for idx in numpy.arange(line.shape[0]):
            for item_id in numpy.arange(self.matrix_ids.items_count):
                self.__item_dist(item_id,line[idx,:],dist)
                pass
            item_ids[idx] = numpy.argmin(dist)
            pass
        return item_ids
    
    def __save_line(self,line,user_id,file_name):
        item_ids = self.__get_line_items(line)
        file_name = file_name % (user_id)
        result = open(file_name,"wt")
        result.write("user id = %d\n" % (user_id,))
        movies = [(item_ids[indice]+1
                   ,self.movies_cvs[self.movies_cvs["id"]==(item_ids[indice]+1)]["name"].values[0]
                   ,self.movies_cvs[self.movies_cvs["id"]==(item_ids[indice]+1)]["gender"].values[0]
                   ) for indice in numpy.arange(item_ids.shape[0]) if not self.movies_cvs[self.movies_cvs["id"]==(item_ids[indice]+1)].empty
                ]
        for movie_info in movies:
            result.write("%d\t%s\t(%s)\n" % movie_info)
            pass
        return 
    
    def __find_best_movies(self,line,movies_cnt):
        return
    
    def __get_rates(self,user_id,movies):
        return
    
    def __save_movies(self,movies,user_id,rates,file_name):
        return
    
    def build_line_for_rand_user(self,rating_info,user_ids,consts):
        for user_id1 in user_ids:
            #user_id1 = self.rng.randint(low=0,high=self.matrix_ids.users_count)
            sys.stdout.write("processing %d user_id " % (user_id1,))
            line = self.__calc_line(user_id1,rating_info)
            self.__save_line(line,user_id1,consts.user_line_file_name)
            movies = self.__find_best_movies(line,100)
            rates = self.__get_rates(user_id1,movies)
            self.__save_movies(movies,user_id1,rates,consts.best_movies_for_user_file_name)
            sys.stdout.write("-- done\n")
            pass 
        return
    
    def __get_rates_of_user(self,user_id,rating_info):
        userids_itemids = numpy.zeros((self.matrix_ids.items_count,2),dtype=theano.config.floatX);
        rates_info = numpy.zeros((self.matrix_ids.items_count,self.rates_approx.ratings_by_user.shape[1] - 1),dtype=theano.config.floatX)
        i0 = 0
        for indice in numpy.arange(self.matrix_ids.items_count):
            userids_itemids[i0,self.rates_approx.user_indice] = user_id
            userids_itemids[i0,self.rates_approx.movie_indice] = indice
            rates_info[i0,:] = rating_info  
            i0 += 1
            pass
        rates = self.rates_approx.get_rates(userids_itemids = userids_itemids, ratesinfo = rates_info)
        return rates
    
    def __save_movie_rates(self,user_id,rates,file_name):
        file_name = file_name % (user_id)
        result = open(file_name,"wt")
        for item_id in numpy.arange(self.matrix_ids.items_count):
            result.write("%d\t" % (item_id,))
            for j in numpy.arange(self.matrix_ids.item_id_size):
                result.write("%f\t" % (self.matrix_ids.items_ids[item_id,j],))
                pass 
            result.write("%f\n" % (rates[item_id],))
            pass
        return
    
    def __save_movies_by_rates(self,user_id,rates,file_name):
        file_name = file_name % (user_id)
        result = open(file_name,"wt")
        indices = numpy.argsort(rates)
        movies = [(indice + 1
                   ,self.movies_cvs[self.movies_cvs["id"]==(indice+1)]["name"].values[0]
                   ,self.movies_cvs[self.movies_cvs["id"]==(indice+1)]["gender"].values[0]
                   ) for indice in indices[-500:] if not self.movies_cvs[self.movies_cvs["id"]==(indice+1)].empty
                ]
        movies.reverse()
        for movie_info in movies:
            result.write("%d -- %s (%s)\n" % movie_info)
            pass
        return
    
    def build_rate_for_rand_user(self,rating_info,user_ids,consts):
        for user_id1 in user_ids:
            #user_id1 = self.rng.randint(low=0,high=self.matrix_ids.users_count)
            sys.stdout.write("processing %d user_id " % (user_id1,))
            rates = self.__get_rates_of_user(user_id1,rating_info)
            self.__save_movie_rates(user_id1,rates,consts.user_rates_of_movies_file_name)
            self.__save_movies_by_rates(user_id1,rates,consts.user_movies_by_rates_file_name)
            sys.stdout.write("-- done\n")
            pass
        return
    
    @staticmethod
    def main(load_id):
        consts = Consts()
        consts.load_from_ids = load_id
        rng = numpy.random.RandomState()
        theano_rng = RandomStreams(rng.randint(2 ** 30))
        user_lines = UserLines(rng = rng,theano_rng = theano_rng,consts = consts)
        rating_info = numpy.zeros(1,dtype=theano.config.floatX)
        wday = 4 # friday
        rating_info[0] = get_aranged(value = wday, min_value = 0, max_value = 6)
        #user_id = user_lines.rng.randint(low=0,high=user_lines.matrix_ids.users_count)
        #user_ids = user_lines.__find_nearest(user_id,5)
        user_ids = [user_lines.rng.randint(low=0,high=user_lines.matrix_ids.users_count) for it in numpy.arange(5)]
        user_lines.build_line_for_rand_user(rating_info = rating_info, user_ids = user_ids, consts = consts)
        user_lines.build_rate_for_rand_user(rating_info = rating_info, user_ids = user_ids, consts = consts)
        sys.stdout.write("all done\n")
        return
    
    pass #class UserLines
     
        
def test_001():
    '''
    functional tests for classes 
    '''
    rng = numpy.random.RandomState()
    theano_rng = RandomStreams(rng.randint(2 ** 30))
    mini_batch_size = 5
    item_size = 7
    user_size = 7
    learning_rate = 0.1
    corruption_level = 0.3
    

    def func(a,b,t):
        return (numpy.sin((2.0*numpy.pi)*(t+a)) + numpy.sin((2.0*numpy.pi)*(t+b)))*0.5    
    #autoencoder.print_state()
    
    autoencoder = AutoEncoder(
        mini_batch_size = mini_batch_size,
        input_size = (item_size+1)*5,encoded_size = user_size,
        hidden_count = 4,hidden_size = 64,activation = T.nnet.relu,
        L1_decay = 0,L2_decay = 0.0, 
        numpy_rng = rng,
        theano_rng = theano_rng,
        )
    repeate_times = 300000
    loss = 0
    t1 = time.clock()
    for i in numpy.arange(repeate_times):
        x_value = numpy.zeros((mini_batch_size,(item_size+1)*5), dtype=theano.config.floatX)
        for j in numpy.arange(mini_batch_size):
            for k in numpy.arange(5):
                a = rng.uniform(0,1)
                b = rng.uniform(0,1)
                t = numpy.arange(float(item_size+1))/float(item_size+1)
                x_value[j,k*(item_size+1):(k+1)*(item_size+1)] = func(a,b,t)
        c1 = autoencoder.train_fn(x_value,learning_rate,corruption_level)
        if numpy.isnan(c1):
            break
        if loss==0:
            loss = c1[0]
        else:
            loss = loss + (c1[0] - loss)*0.001
        if i%1000 == 0:
            print("%d -- %f" % (i/1000,loss))
    t2 = time.clock()
    print("train for : %f sec, times : %d" % ((t2-t1)/repeate_times,repeate_times))
    #print(c)
    #autoencoder.print_state()
    repeate_times = 1000
    t1 = time.clock()
    for i in numpy.arange(repeate_times) :
        x_value = numpy.asarray(
            rng.uniform(
                low = -0.5, 
                high = 0.5, 
                size = (mini_batch_size,(item_size+1)*5)))
        encoded_value = autoencoder.get_encoded_fn(x_value)
    t2 = time.clock()
    print("get_encoded for : %f sec" % ((t2-t1)/repeate_times))
    print(encoded_value[0]+0.5)
    
    #print("function autoencoder.train :")
    #theano.printing.debugprint(autoencoder.train)
    
    #print("function autoencoder.__get_encoded :")
    #theano.printing.debugprint(autoencoder.__get_encoded)
    
    return
    
def test_002():
    rng = numpy.random.RandomState()
    theano_rng = RandomStreams(rng.randint(2 ** 30))
    batch_size = 5
    input_size = 10
    output_size = 2
    learning_rate = 0.1
    
    net1 = ApproxNet(
        batch_size = batch_size
        ,input_size = output_size
        ,output_size = input_size
        ,hidden_count = 6,hidden_size = 64,hidden_activation = T.nnet.relu
        ,numpy_rng = rng
        ,theano_rng = theano_rng
        ,L1_decay = 0
        ,L2_decay = 0
        )
    
    def func(a,b,t):
        return (numpy.sin((2.0*numpy.pi)*(t+a)) + numpy.sin((2.0*numpy.pi)*(t+b)))*0.5    

    #c = []
    repeate_times = 100000
    loss = 0
    t1 = time.clock()
    for i in numpy.arange(repeate_times):
        x_value = numpy.zeros((batch_size,input_size), dtype=theano.config.floatX)
        y_value = numpy.zeros((batch_size,output_size),dtype=theano.config.floatX)
        for j in numpy.arange(batch_size):
            a = rng.uniform(0.0,1.0)
            b = rng.uniform(0.0,1.0)
            t = numpy.arange(float(input_size))/float(input_size)
            x_value[j,0:input_size] = func(a,b,t)
            y_value[j,0] = a - 0.5
            y_value[j,1] = b - 0.5 
        c1 = net1.train_fn(y_value,x_value,learning_rate)
        if loss == 0:
            loss = c1[0]
        else:
            loss = loss + (c1[0]-loss)*0.001
        #c.append(c1)
        if numpy.isnan(c1):
            break
        if i % 1000 == 0:
            print("%d -- %f" % (i/1000,loss))
    t2 = time.clock()
    print("train for : %f sec, times : %d" % ((t2-t1)/repeate_times,repeate_times))
    #print(c)
    
    x_value = numpy.zeros((batch_size,input_size), dtype=theano.config.floatX)
    y_value = numpy.zeros((batch_size,output_size),dtype=theano.config.floatX)
    for j in numpy.arange(batch_size):
        a = rng.uniform(0.0,1.0)
        b = rng.uniform(0.0,1.0)
        t = numpy.arange(float(input_size))/float(input_size)
        x_value[j,0:input_size] = func(a,b,t)
        y_value[j,0] = a - 0.5
        y_value[j,1] = b - 0.5
    x_result = net1.run_fn(y_value)
    print(x_value+0.5)
    print(x_result[0]+0.5)
 
    return

def prepare_data():
    consts = Consts()
    rng = numpy.random.RandomState()
    theano_rng = RandomStreams(rng.randint(2 ** 30))
    rs = RecommenderSystem(rng= rng,theano_rng = theano_rng,consts=consts)
    rs.prepare_data(consts)
    return

def trace(index,loss_items_to_user,loss_users_to_item,loss_itemids,loss_rates,validate_loss,validate_loss_min,trace_file_name):
    tf = open(trace_file_name,"at")
    tf.write("%d\t%f\t%f\t%f\t%f\t%f\t%f\n" % (index,loss_items_to_user,loss_users_to_item,loss_itemids,loss_rates,validate_loss,validate_loss_min)) 
    return

def train_all():
    consts = Consts()
    rng = numpy.random.RandomState()
    theano_rng = RandomStreams(rng.randint(2 ** 30))
    rs = RecommenderSystem(rng= rng,theano_rng = theano_rng,consts=consts)
    
    #print("userids, shape : " + str(rs.usersids.shape))
    #print(rs.usersids)
    #print("moviesids, shape : " + str(rs.moviesids.shape))
    #print(rs.moviesids)
    #print("ratings_by_user, shape : " + str(rs.ratings_by_user.shape))
    #print(rs.ratings_by_user)
    #print("ratings_by_user_ids, shape : " + str(rs.ratings_by_user_ids.shape))
    #print(rs.ratings_by_user_ids)
    #print("ratings_by_user_idx, shape : " + str(rs.ratings_by_user_idx.shape))
    #print(rs.ratings_by_user_idx)
    #print("ratings_by_movie, shape : " + str(rs.ratings_by_movie.shape))
    #print(rs.ratings_by_movie)
    #print("ratings_by_movie_ids, shape : " + str(rs.ratings_by_movie_ids.shape))
    #print(rs.ratings_by_movie_ids)
    #print("atings_by_movie_idx, shape : " + str(rs.ratings_by_movie_idx.shape))
    #print(rs.ratings_by_movie_idx)
    
    sys.stdout.write("i item2user user2item  itemids    rates    rval  rvalmin\n")
    
    loss_rates = float(0)
    loss_items_to_user = float(0)
    loss_users_to_item = float(0)
    validate_loss_min = float(0)
    validate_loss = float(0)
    loss_itemids = float(0)
    for idx in numpy.arange(100000):
        lt = time.time()
        for j in numpy.arange(consts.ids_move_count):
            loss_items_to_user,loss_users_to_item = rs.train_encoders(learning_rate = consts.encoder_learning_rate, corruption_level = consts.encoder_corruption_rate,consts=consts)
            if rng.rand(1)[0]<=consts.train_rates_rate:
                loss_rates = rs.train_rates(learning_rate = consts.result_learning_rate)
            if rng.rand(1)[0]<consts.train_itemids_rate:
                loss_itemids = rs.train_itemids(learning_rate = consts.itemids_learning_rate, consts = consts)
            t1 = time.time()
            if t1>lt+1:
                sys.stdout.write("\t\t\t\t\t\t\t\t\t\r")
                sys.stdout.write("[%d] %f %f %f %f %f %f\r" % (idx,loss_items_to_user,loss_users_to_item,loss_itemids,loss_rates,validate_loss,validate_loss_min))
                lt = lt+1
            pass
        rs.calc_new_ids(consts=consts)
        if idx % consts.save_cycles == 0:
            rs.save((idx/consts.save_cycles) + consts.load_from_ids,consts) 
            rs.save_rates((idx/consts.save_cycles) + consts.load_from_ids,consts)
        if idx % consts.validate_cycles == 0:
            validate_loss = rs.validate_rates(consts=consts)
            if validate_loss_min==0 or validate_loss<validate_loss_min:
                validate_loss_min = validate_loss
                rs.save_rates(0,consts)
        consts.update_index(idx + (consts.load_from_ids*consts.save_cycles))
        trace(idx + (consts.load_from_ids*consts.save_cycles),loss_items_to_user,loss_users_to_item,loss_itemids,loss_rates,validate_loss,validate_loss_min,consts.trace_file_name)
        pass
        
    return

def trace_rates(index,loss_rate,validate_loss_min,validate_loss,trace_file_name):
    tf = open(trace_file_name,"at")
    tf.write("%d\t%f\t%f\t%f\n" % (index,loss_rate,validate_loss,validate_loss_min)) 
    return

def train_rates():
    consts = Consts()
    rng = numpy.random.RandomState()
    theano_rng = RandomStreams(rng.randint(2 ** 30))
    rs = RecommenderSystem(rng= rng,theano_rng = theano_rng,consts=consts)
    validate_loss_min = 0
    validate_loss = 0
    for i in numpy.arange(100000):
        lt = time.time()
        for j in numpy.arange(consts.ids_move_count):
            loss_rates = rs.train_rates(learning_rate = consts.result_learning_rate)
            t1 = time.time()
            if t1>lt+1:
                sys.stdout.write("\t\t\t\t\t\t\t\t\t\r")
                sys.stdout.write("[%d] loss = %f , val = %f valmin = %f\r" % (i,loss_rates,validate_loss,validate_loss_min))
                lt = lt+1
        trace_rates(i + (consts.load_from_ids*consts.save_cycles),loss_rates,validate_loss_min,validate_loss,consts.trace_rates_file_name)
        if i % consts.save_cycles == 0:
            rs.save_rates((i/consts.save_cycles) + consts.load_from_ids,consts)
        if i % consts.validate_cycles == 0:
            validate_loss = rs.validate_rates(consts=consts)
            if validate_loss_min==0 or validate_loss<validate_loss_min:
                validate_loss_min = validate_loss
                rs.save_rates(0,consts)
        consts.update_index(i + (consts.load_from_ids*consts.save_cycles))
        
    return

def nearest_movies(indexes):
    print("finding nearest ...")
    #index = 49
    #indexes = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20]
    #movie_ids = [14,1251,2132,1974,306,3742,3415,3224,2946,3503]
    movie_ids = [1251] #,1974
    consts = Consts()
    for index in indexes:
        nm = NearestMovies(index,consts)
        for movie_id in movie_ids:
            nearest_movies = nm.get_nearest_movies(movie_id=movie_id)
            filename = consts.nearest_movies_file_name % (movie_id,index)
            outfile = open(filename,"wt")
            i = 0
            for movie_info in nearest_movies:
                print("[%d] %f -- %d : %s -- %s" % (i,movie_info[0],movie_info[1],movie_info[2],movie_info[3]))
                outfile.write("[%d] %f -- %d : %s -- %s\n" % (i,movie_info[0],movie_info[1],movie_info[2],movie_info[3]))
                i = i + 1
    print("done.")
    return

def get_clusters(index):
    print("getting clusters ...")
    consts = Consts()
    nm = NearestMovies(index,consts)
    res = nm.get_clusters(cluster_number = 100, iterations = 500)
    filename = consts.knearest_movies_file_name % (index)
    outfile = open(filename,"wt")
    gi = 0
    for r in res:
        outfile.write("group %d\n" % (gi,))
        for movie_info in r:
            outfile.write("%d : %s -- %s\n" % (movie_info[0],movie_info[1],movie_info[2])) 
        gi = gi + 1
		pass
    print("clusters was saved to the file.")
    return

def convert_to_dta(indexes):
    print("converting npy to dta ...")
    #indexes = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17]
    consts = Consts()
    for index in indexes:
        nm = NearestMovies(index,consts)
        nm.save_dta(index = index, consts = consts)
		pass
    print("done.")
    return    

def test_005():
    consts = Consts()
    for i in numpy.arange(1000):
        consts.update_index(i)
        print(consts.ids_move_count_coef)
    return

if __name__ == '__main__':
    print("Python version : " + str(sys.version))
    print("Numpy version : " + str(numpy.__version__))
    print("Pandas version : " + str(pandas.__version__))
    print("Theano version : " + str(theano.__version__))
    
    #prepare_data()
    
    #test_001()
    #test_002()
    #test_003()
    #test_005()

    train_mode = True
    user_lines_mode = False
        
    if train_mode:
        train_all()
        pass
    else:
        if user_lines_mode:
            UserLines.main(0)
            pass
        else:
            indexes = [1]
            nearest_movies(indexes = indexes)
            convert_to_dta(indexes = indexes)
            pass
        pass
    
    #get_clusters(410)
    
