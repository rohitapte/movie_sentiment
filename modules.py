import tensorflow as tf
from tensorflow.python.ops import rnn_cell

class NeuralNetworkHiddenLayer(object):
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
