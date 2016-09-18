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
import numpy

import theano
import theano.tensor as T
from theano.tensor.shared_randomstreams import RandomStreams


def save_layer(file_name,obj_name,value,consts):
    #file_name = consts.get_file_name_by_index(indx,file_name)
    file_name = file_name % (obj_name,) 
    numpy.save(file = file_name, arr = value)
    return

def load_layer(file_name,obj_name,consts):
    #file_name = consts.get_file_name_by_index(indx,file_name)
    file_name = file_name % (obj_name,)
    if not os.path.isfile(path = file_name):
        return None 
    return numpy.asarray(a = numpy.load(file = file_name),dtype=theano.config.floatX)

class ApproxNet(object):
    '''
    The deep net for regression
    '''
    def __create_layer(self, numpy_rng, batch_size, layer_size, W, b, prev_layer, i):
        if not W or not W[i]:
            delta = numpy.sqrt(6 / (float(prev_layer) + float(layer_size)))
            initial_W = numpy.asarray(
                numpy_rng.uniform(
                    low = -delta, 
                    high = delta, 
                    size = (prev_layer, layer_size)))
            self.W.append(theano.shared(value = initial_W, name = 'W' + str(i)))
            #print("W%d size = (%d,%d)" % (i,prev_layer, layer_size))
        else:
            self.W.append(W[i])
            
        if not b or not b[i]:
            self.b.append(theano.shared(value = numpy.zeros(layer_size, dtype=theano.config.floatX),name = 'b'+str(i)))
            #print("b%d size = (%d,%d)" % (i,1,layer_size))
        else:
            self.b.append(b[i])
            
        self.Result.append(theano.shared(value = numpy.zeros((batch_size,layer_size), dtype=theano.config.floatX),name = 'Result'+str(i)))
        #print("Result%d size = (%d,%d)" % (i,batch_size,layer_size))
        return layer_size
    
    def __create_hidden_layers(self, numpy_rng, batch_size, hidden_count, hidden_size, W, b, prev_layer,base_i):
        for i in numpy.arange(hidden_count):
            prev_layer = self.__create_layer(numpy_rng, batch_size, hidden_size, W, b, prev_layer, base_i+i)
        return prev_layer    

    def __get_processed(self, input_x):
        """ 
        Computes the values of the encoded layer 
        """
        data = input_x
        for idx in numpy.arange(self.hidden_count):
            self.Result[idx] = self.hidden_activation(T.dot(data, self.W[idx]) + self.b[idx])
            data = self.Result[idx] 
        self.Result[self.hidden_count] = T.tanh(T.dot(data, self.W[self.hidden_count]) + self.b[self.hidden_count])
        return self.Result[self.hidden_count]

    def __get_L1(self):
        self.L1 = 0
        if len(self.W)==0:
            return self.L2
        for W in self.W:
            self.L1 = self.L1 + T.mean(T.abs_(W))
        return self.L1/len(self.W)
    
    def __get_L2(self):
        self.L2 = 0
        if len(self.W)==0:
            return self.L2
        for W in self.W:
            self.L2 = self.L2 + T.mean(T.sqr(W))
        return self.L2/len(self.W)
    
    def __get_cost_updates(self, target,learning_rate,L1_decay,L2_decay):
        """ This function computes the cost and the updates for one trainng
        step of the dA """

        y = self.__get_processed(self.input_x)
        # note : we sum over the size of a datapoint; if we are using
        #        minibatches, L will be a vector, with one entry per
        #        example in minibatch
        L = T.mean(T.sqr(y-target),axis=1)
        # note : L is now a vector, where each element is the
        #        cross-entropy cost of the reconstruction of the
        #        corresponding example of the minibatch. We need to
        #        compute the average of all these to get the cost of
        #        the minibatch
        cost = T.mean(L) + self.__get_L2() * L2_decay + self.__get_L1() * L1_decay

        # compute the gradients of the cost of the `dA` with respect
        # to its parameters
        gparams = T.grad(cost, self.params)
        # generate the list of updates
        updates = []
        updates.extend([
            (param, param - learning_rate * gparam)
            for param, gparam in zip(self.params, gparams)
            ])

        return (cost, updates)
    
    def __get_run(self):
        return self.__get_processed(self.input_x)
    
    def __init__(self
                ,batch_size
                ,input_size
                ,output_size
                ,hidden_count,hidden_size,hidden_activation
                ,numpy_rng
                ,theano_rng = None
                ,L1_decay = 0
                ,L2_decay = 0
                ,W = None
                ,b = None
                ,input_x = None
                ,target_y = None
                ,result_y = None
                ):
        
        
        if not theano_rng:
            theano_rng = RandomStreams(numpy_rng.randint(2 ** 30))
        
        self.theano_rng = theano_rng
        self.input_size = input_size
        self.output_size = output_size
        self.hidden_count = hidden_count
        self.hiden_size = hidden_size
        self.hidden_activation = hidden_activation
        if not input_x:
            input_x = T.matrix(name="x",dtype=theano.config.floatX)
        if not target_y:
            target_y = T.matrix(name="target",dtype=theano.config.floatX)
        if not result_y:
            result_y = T.matrix(name="y",dtype=theano.config.floatX)
            
        self.input_x = input_x
        self.target_y = target_y
        self.result_y = result_y

        self.W = []
        self.b = []
        self.Result = []
        
        prev_layer = input_size        
        prev_layer = self.__create_hidden_layers(numpy_rng, batch_size, hidden_count, hidden_size, W, b, prev_layer,0)
        prev_layer = self.__create_layer(numpy_rng, batch_size, output_size, W, b, prev_layer, hidden_count)            
        
        self.params = []
        self.params.extend(self.W)
        self.params.extend(self.b)
        
        self.learning_rate = T.scalar(name = "learning_rate",dtype=theano.config.floatX)
        self.L1 = T.scalar(name = "L1",dtype=theano.config.floatX)
        self.L2 = T.scalar(name = "L2",dtype=theano.config.floatX)
        
        # create functions of deep net 
        cost,updates = self.__get_cost_updates(target = self.target_y, learning_rate = self.learning_rate,L1_decay = L1_decay,L2_decay = L2_decay)
        self.train_fn = theano.function(inputs = [self.input_x,self.target_y,self.learning_rate],outputs = [cost],updates=updates)
        self.result_y = self.__get_run()
        self.run_fn = theano.function(inputs=[self.input_x],outputs=[self.result_y])
        
        return
    
    def save_state(self,file_name,consts):
        i = 0;
        for W in self.W:
            save_layer(file_name,"W"+str(i),W.get_value(),consts)
            i=i+1
        i = 0
        for b in self.b:
            save_layer(file_name,"b" + str(i),b.get_value(),consts)
            i=i+1
        return
    
    def load_state(self,file_name,consts):
        i = 0;
        for W in self.W:
            layer = load_layer(file_name,"W"+str(i),consts)
            if layer is None:
                return False
            W.set_value(layer)
            i=i+1
        i = 0
        for b in self.b:
            layer = load_layer(file_name,"b" + str(i),consts)
            if layer is None:
                return False
            b.set_value(layer)
            i=i+1
        return True
    
    def print_state(self):
        i = 0;
        for W in self.W:
            print("W"+str(i));
            print(W.get_value())
            i=i+1
        i = 0
        for b in self.b:
            print("b" + str(i))
            print(b.get_value())
        #i = 0
        #for result in self.Result:
        #    print("Result"+str(i))
        #    print(result.get_value())
        return
    

class AutoEncoder(object):
    '''
    The auto encoder deep net. 
    '''
    
    def __create_layer(self, numpy_rng, mini_batch_size, layer_size, W, b, prev_layer, i):
        if not W or not W[i]:
            delta = numpy.sqrt(6 / (float(prev_layer) + float(layer_size)))
            initial_W = numpy.asarray(
                numpy_rng.uniform(
                    low = -delta, 
                    high = delta, 
                    size = (prev_layer, layer_size))
                ,dtype=theano.config.floatX
                )
            self.W.append(theano.shared(value = initial_W, name = 'W' + str(i)))
            #print("W%d size = (%d,%d)" % (i,prev_layer, layer_size))
        else:
            self.W.append(W[i])
        if not b or not b[i]:
            self.b.append(theano.shared(value = numpy.zeros(layer_size, dtype=theano.config.floatX),name = 'b'+str(i)))
            #print("b%d size = (%d,%d)" % (i,1,layer_size))
        else:
            self.b.append(b[i])
        self.Result.append(theano.shared(value = numpy.zeros((mini_batch_size,layer_size), dtype=theano.config.floatX),name = 'Result'+str(i)))
        #print("Result%d size = (%d,%d)" % (i,mini_batch_size,layer_size))
        return layer_size

    def __create_hidden_layers(self, numpy_rng, mini_batch_size, hidden_count, hidden_size, W, b, prev_layer,base_i):
        for i in numpy.arange(hidden_count):
            prev_layer = self.__create_layer(numpy_rng, mini_batch_size, hidden_size, W, b, prev_layer, base_i+i)
        return prev_layer    

    def __get_corrupted_input(self, input_x, corruption_level):
        """This function keeps ``1-corruption_level`` entries of the inputs the
        same and zero-out randomly selected subset of size ``coruption_level``
        Note : first argument of theano.rng.binomial is the shape(size) of
               random numbers that it should produce
               second argument is the number of trials
               third argument is the probability of success of any trial

                this will produce an array of 0s and 1s where 1 has a
                probability of 1 - ``corruption_level`` and 0 with
                ``corruption_level``

                The binomial function return int64 data type by
                default.  int64 multiplicated by the input
                type(floatX) always return float64.  To keep all data
                in floatX when floatX is float32, we set the dtype of
                the binomial to floatX. As in our case the value of
                the binomial is always 0 or 1, this don't change the
                result. This is needed to allow the gpu to work
                correctly as it only support float32 for now.

        """
        return self.theano_rng.binomial(
                size=input_x.shape, n=1,
                p= 1 - corruption_level,
                dtype=theano.config.floatX) * input_x

    def __get_encoded(self, input_x):
        """ 
        Computes the values of the encoded layer 
        """
        data = input_x
        for idx in numpy.arange(self.hidden_count):
            self.Result[idx] = self.activation(T.dot(data, self.W[idx]) + self.b[idx])
            data = self.Result[idx]
        self.Result[self.hidden_count] = T.tanh(T.dot(data, self.W[self.hidden_count]) + self.b[self.hidden_count])*float(0.5)
        return self.Result[self.hidden_count]

    def __get_reconstructed(self,encoded):
        """
        Computes the values of the result layer 
        """
        data = encoded
        base_i = self.hidden_count+1
        for idx in numpy.arange(self.hidden_count):
            self.Result[base_i+idx] = self.activation(T.dot(data, self.W[base_i+idx]) + self.b[base_i+idx])
            data = self.Result[base_i+idx]
        self.Result[base_i+self.hidden_count] = T.tanh(T.dot(data, self.W[base_i+self.hidden_count]) + self.b[base_i+self.hidden_count])
        return self.Result[base_i+self.hidden_count]
    
    def __get_L1(self):
        self.L1 = 0
        if len(self.W)==0:
            return self.L2
        for W in self.W:
            self.L1 = self.L1 + T.mean(T.abs_(W))
        return self.L1/len(self.W)
    
    def __get_L2(self):
        self.L2 = 0
        if len(self.W)==0:
            return self.L2
        for W in self.W:
            self.L2 = self.L2 + T.mean(T.sqr(W))
        return self.L2/len(self.W)
        
    def __get_cost_updates(self, corruption_level, learning_rate,L1_decay,L2_decay):
        """ This function computes the cost and the updates for one trainng
        step of the dA """

        tilde_x = self.__get_corrupted_input(self.input_x, corruption_level)
        y = self.__get_encoded(tilde_x)
        z = self.__get_reconstructed(y)
        # note : we sum over the size of a datapoint; if we are using
        #        minibatches, L will be a vector, with one entry per
        #        example in minibatch
        #L = - T.sum(self.input_x * T.log(z) + (1 - self.input_x) * T.log(1 - z), axis=1)
        L = T.mean(T.sqr(z-self.input_x),axis=1)
        # note : L is now a vector, where each element is the
        #        cross-entropy cost of the reconstruction of the
        #        corresponding example of the minibatch. We need to
        #        compute the average of all these to get the cost of
        #        the minibatch
        cost = T.mean(L) + self.__get_L2() * L2_decay + self.__get_L1() * L1_decay

        # compute the gradients of the cost of the `dA` with respect
        # to its parameters
        gparams = T.grad(cost, self.params)
        # generate the list of updates
        updates = []
        updates.extend([
            (param, param - learning_rate * gparam)
            for param, gparam in zip(self.params, gparams)
            ])

        return (cost, updates)
    
    def __get_run(self):
        return self.__get_encoded(self.input_x)

    def __init__(self,
        mini_batch_size,
        input_size,encoded_size,
        hidden_count,hidden_size,activation,
        L1_decay,L2_decay,
        numpy_rng,
        theano_rng = None,
        W = None,
        b = None,
        input_x = None
        ):
        '''
        Constructor
        '''
        
        if not theano_rng:
            theano_rng = RandomStreams(numpy_rng.randint(2 ** 30))
        
        self.theano_rng = theano_rng
        self.input_size = input_size
        self.encoded_size = encoded_size
        self.hidden_count = hidden_count
        self.hiden_size = hidden_size
        self.activation = activation
        if not input_x:
            input_x = T.matrix(name="x",dtype=theano.config.floatX)
            
        self.input_x = input_x

        self.W = []
        self.b = []
        self.Result = []
        
        prev_layer = input_size        
        prev_layer = self.__create_hidden_layers(numpy_rng, mini_batch_size, hidden_count, hidden_size, W, b, prev_layer,0)
        prev_layer = self.__create_layer(numpy_rng, mini_batch_size, encoded_size, W, b, prev_layer, hidden_count)            
        prev_layer = self.__create_hidden_layers(numpy_rng, mini_batch_size, hidden_count, hidden_size, W, b, prev_layer,hidden_count+1)
        prev_layer = self.__create_layer(numpy_rng, mini_batch_size, input_size, W, b, prev_layer, 2*hidden_count+1)
        
        self.params = []
        self.params.extend(self.W)
        self.params.extend(self.b)
        
        self.learning_rate = T.scalar(name = "learning_rate",dtype=theano.config.floatX)
        self.corruption_level = T.scalar(name = "learning_rate",dtype=theano.config.floatX)
        self.L1 = T.scalar(name = "L1",dtype=theano.config.floatX)
        self.L2 = T.scalar(name = "L2",dtype=theano.config.floatX)
        
        # create functions of autoencoder
        cost,updates = self.__get_cost_updates(corruption_level = self.corruption_level, learning_rate = self.learning_rate,L1_decay = L1_decay,L2_decay = L2_decay)
        self.train_fn = theano.function(inputs = [self.input_x,self.learning_rate,self.corruption_level],outputs = [cost],updates=updates)
        self.encoded = self.__get_run()
        self.get_encoded_fn = theano.function(inputs=[self.input_x],outputs=[self.encoded])
        
        return
    
    def save_state(self,file_name,consts):
        i = 0;
        for W in self.W:
            save_layer(file_name,"W"+str(i),W.get_value(),consts)
            i=i+1
        i = 0
        for b in self.b:
            save_layer(file_name,"b" + str(i),b.get_value(),consts)
            i=i+1
        return
    
    def load_state(self,file_name,consts):
        i = 0;
        for W in self.W:
            layer = load_layer(file_name,"W"+str(i),consts)
            if layer is None:
                return False
            W.set_value(layer)
            i=i+1
        i = 0
        for b in self.b:
            layer = load_layer(file_name,"b" + str(i),consts)
            if layer is None:
                return False
            b.set_value(layer)
            i=i+1
        return True
    
    def print_state(self):
        i = 0;
        for W in self.W:
            print("W"+str(i));
            print(W.get_value())
            i=i+1
        i = 0
        for b in self.b:
            print("b" + str(i))
            print(b.get_value())
        #i = 0
        #for result in self.Result:
        #    print("Result"+str(i))
        #    print(result.get_value())
        return
