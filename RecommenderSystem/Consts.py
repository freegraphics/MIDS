'''
Created on 16.08.10

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

import os

import theano
import theano.tensor as T
from theano.tensor.shared_randomstreams import RandomStreams

class Consts(object):
    '''
    default constants of classes 
    '''
    
    def __init__(self):
        '''
        the constructor
        '''
        
        self.load_from_ids = int(12) 
        
        # rates constants
        self.MaxRate = int(5)
        
        # paths constants
        self.data_path = "F:\\works\\projects\\python\\RecommenderSystem\\data" 
        self.result_path = "F:\\works\\projects\\python\\RecommenderSystem\\tests\\16.09.24 -- 17-35"
        self.trained_path = "F:\\works\\projects\\python\\RecommenderSystem\\tests\\16.09.24 -- 17-35\\ids_000"

        # source data file names        
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
        
        # result data file names
        self.users_ids_file_name = os.path.join(self.result_path,"users_ids.npy")
        self.items_ids_file_name = os.path.join(self.result_path,"items_ids.npy")
        self.nearest_movies_file_name = os.path.join(self.result_path,"nearest_movies_%d_%03d.txt")
        self.knearest_movies_file_name = os.path.join(self.result_path,"knearest_movies_%03d.txt")
        self.users_ids_to_item_id_autoencoder_file_name = os.path.join(self.result_path,"users_ids_ldt_%s.npy")
        self.items_ids_to_user_id_autoencoder_file_name = os.path.join(self.result_path,"items_ids_ldt_%s.npy")
        self.user_ids_rate_to_item_ids_net_file_name = os.path.join(self.result_path,"user_ids_rate_to_item_ids_ldt_%s.npy")
        self.result_net_file_name = os.path.join(self.result_path,"result_ldt_%s.npy")
        self.users_ids_dta_file_name = os.path.join(self.result_path,"users_ids.dta")
        self.items_ids_dta_file_name = os.path.join(self.result_path,"items_ids.dta")
        self.trace_file_name = os.path.join(self.result_path,"trace.txt")
        self.trace_rates_file_name = os.path.join(self.result_path,"trace_rates.txt")
        self.user_line_file_name = os.path.join(self.result_path,"%04d_line.txt")
        self.best_movies_for_user_file_name = os.path.join(self.result_path,"%04d_best_movies.txt")
        self.user_rates_of_movies_file_name = os.path.join(self.result_path,"%04d_movies_rate.txt")
        self.user_movies_by_rates_file_name = os.path.join(self.result_path,"%04d_movies_by_rate.txt")
        
        self.save_cycles = int(50) 
    
        # MIDS constants
        self.user_id_size = int(31)
        self.item_id_size = int(31)
        
        self.user_max_distance = float(256.0*self.user_id_size)
        self.item_max_distance = float(256.0*self.item_id_size)
        
        self.encode_elements_count = int(5)
        
        # encoder defaults 
        self.encoder_batch_size = int(32)
        self.encoder_learning_rate = float(0.1)
        self.encoder_corruption_rate = float(0.0)
        self.encoder_hidden_layers_count = int(6)
        self.encoder_hidden_layers_activation = T.nnet.relu
        self.encoder_hidden_layer_size = int(256)    
        self.encoder_L1_decay = float(0.0)
        self.encoder_L2_decay = float(1.0e-4)
        self.encoder_loss_k = float(1.0e-3)
        
        # result defaults
        self.result_batch_size = int(32)
        self.result_learning_rate = float(0.1)
        self.result_hidden_layers_count = int(6)
        self.result_hidden_layers_activation = T.nnet.relu
        self.result_hidden_layer_size = int(256)    
        self.result_L1_decay = float(0.0)
        self.result_L2_decay = float(1.0e-4)
        self.result_loss_k = float(1.0e-3)
        
        # items ids net  defaults
        self.itemids_batch_size = int(32)
        self.itemids_learning_rate = float(0.1)
        self.itemids_hidden_layers_count = int(6)
        self.itemids_hidden_layers_activation = T.nnet.relu
        self.itemids_hidden_layer_size = int(256)
        self.itemids_L1_decay = float(0.0)
        self.itemids_L2_decay = float(1.0e-4)
        self.itemids_loss_k = float(1.0e-3)
        
        self.train_rate = float(0.9)
        self.validate_cycles = int(5)
        
        # move ids constants
        self.users_ids_move_elem_count_rate = float(0.1)
        self.items_ids_move_elem_count_rate = float(0.2)
        self.users_ids_move_elem_count_rate1 = float(0.1)
        self.items_ids_move_elem_count_rate1 = float(0.1)
        self.users_ids_avg_rate = float(0.2)
        self.items_ids_avg_rate = float(0.2)
        self.new_user_cycles = int(100)
        self.new_item_cycles = int(100)
        self.ids_update_users_normilized_vs_avg_rate = float(0.99)
        self.ids_update_items_normilized_vs_avg_rate = float(0.99)
        
        self.avg_dx_item_small_weight = float(0.3)
        self.min_max_compresion_rate = float(0.95)

        self.ids_move_count = int(1000) 
        self.ids_move_count_coef = float(0.5)
        self.dist_sqrt_coef = float(9.0)
        
        self.train_rates_rate = float(0.5)
        self.train_itemids_rate = float(0.5)

        self.update_index(self.load_from_ids*self.save_cycles)
        return
    
    def get_file_name_by_index(self,index,file_name):
        path,fname = os.path.split(file_name)
        path = os.path.join(path,"ids_%03d" % (index))
        if not os.path.exists(path):
            os.mkdir(path)
        return os.path.join(path,fname)
    
    def update_index(self,index):
        if index<int(100):
            self.users_ids_move_elem_count_rate = float(0.2) # 6040*0.25 == 1510
            self.items_ids_move_elem_count_rate = float(0.2) # 3000*0.25 == 750
            self.users_ids_move_elem_count_rate1 = float(0.25)
            self.items_ids_move_elem_count_rate1 = float(0.25)
            self.ids_move_count = int(2000)
            self.ids_move_count_coef = float(0.5)
            return
        if index>=int(100) and index<int(500):
            r = float(index-1000)/float(1500.0-1000.0)
            self.users_ids_move_elem_count_rate = float((1.0-r)*0.2 + r*0.25)
            self.items_ids_move_elem_count_rate = float((1.0-r)*0.2 + r*0.25)
            self.users_ids_move_elem_count_rate1 = float((1.0-r)*0.25 + r*1.0)
            self.items_ids_move_elem_count_rate1 = float((1.0-r)*0.25 + r*1.0)
            self.ids_move_count = int((1.0-r)*2000 + r*2000) 
            self.ids_move_count_coef = float((1.0-r)*0.5 + r*0.5)
            return
        if index>=int(500):
            r = float(index-500)/float(1000.0*3.0)
            self.users_ids_move_elem_count_rate = float(0.25)
            self.items_ids_move_elem_count_rate = float(0.25)
            self.users_ids_move_elem_count_rate1 = float(1.0)
            self.items_ids_move_elem_count_rate1 = float(1.0)
            self.ids_move_count = int(2000) 
            self.ids_move_count_coef = float(0.5/(1.0+r))
            return
        return
