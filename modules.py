import tensorflow as tf
from model import NeuralLayer
#from tensorflow.python.ops import rnn_cell

class NeuralNetworkHiddenLayer(NeuralLayer):
    def __init__(self,name,input_size,hidden_size,keep_prop):
        self.name=name
        self.keep_prob=keep_prop
        self.input_size=input_size
        self.hidden_size=hidden_size

    def build_graph(self,values):
        hidden_weight=tf.get_variable(name=self.name+'_hidden_weight',shape=[self.input_size,self.hidden_size],dtype=tf.float32,initializer=tf.truncated_normal_initializer(stddev=0.05))
        hidden_bias=tf.get_variable(name=self.name+'_hidden_bias',shape=[self.hidden_size],dtype=tf.float32,initializer=tf.constant_initializer(0.0))
        output=tf.nn.relu(tf.matmul(values,hidden_weight)+hidden_bias)
        output=tf.nn.dropout(output,self.keep_prob)
        return output

class Convolution2DLayer(NeuralLayer):
    def __init__(self,name,filter,strides,channel_size,pool_size=None):
        self.name=name
        self.filter=filter
        self.strides=strides
        self.channel_size=channel_size
        self.pool_size=pool_size
        #channel_size must = filter[3]
        #filter[2] must=values[3]

    def build_graph(self,values):   #values has shape (batch,X,Y,Z)
        conv_filter=tf.get_variable(name=self.name+'_conv_filter', shape=self.filter,dtype=tf.float32,initializer=tf.truncated_normal_initializer(stddev=0.05)) #filter is [3,B,Z,channel_size]
        conv=tf.nn.conv2d(values,conv_filter,self.strides,padding='SAME')       #conv has shape (1,X/stride[1],Y/stride[2],channel_size)
        conv_bias=tf.get_variable(name=self.name+'_conv_bias',shape=self.channel_size,dtype=tf.float32,initializer=tf.constant_initializer(0.0))
        conv_bias_add=tf.nn.bias_add(conv,conv_bias)
        conv_relu=tf.nn.relu(conv_bias_add)
        if self.pool_size is not None:
            context_pool=tf.nn.max_pool(conv_relu,ksize=self.pool_size,strides=self.pool_size,padding='SAME',name=self.name+'_max_pool')
            return context_pool     #conv has shape (1,X/pool_size[1],Y/pool_size[2],channel_size)
        else:
            return conv_relu

class Convolution1DLayer(NeuralLayer):
    def __init__(self,name,filter,strides,channel_size,pool_size=None):
        self.name=name
        self.filter=filter
        self.strides=strides
        self.channel_size=channel_size
        self.pool_size=pool_size
        # channel_size must = filter[2]
        # filter[1] must=values[2]

    def build_graph(self,values):   #values has shape (batch,X,Y)
        conv_filter=tf.get_variable(name=self.name + '_conv_filter', shape=self.filter, dtype=tf.float32,initializer=tf.truncated_normal_initializer(stddev=0.05))  # filter is [3,Y,channel_size]
        conv=tf.nn.conv1d(values,conv_filter,self.strides,padding='SAME')  # conv has shape (1,X/stride,channel_size)
        conv_bias=tf.get_variable(name=self.name+'_conv_bias',shape=self.channel_size,dtype=tf.float32,initializer=tf.constant_initializer(0.0))
        conv_bias_add=tf.nn.bias_add(conv,conv_bias)
        conv_relu=tf.nn.relu(conv_bias_add)
        if self.pool_size is not None:
            context_pool=tf.nn.pool(conv_relu,window_shape=self.pool_size,strides=self.pool_size,pooling_type='MAX',padding='SAME',name=self.name+'_max_pool')
            return context_pool     #context_pool has shape(1,conv_relu[1]/pool_size,channels)
        else:
            return conv_relu  # conv has shape (1,X/pool_size[1],Y/pool_size[2],channel_size)