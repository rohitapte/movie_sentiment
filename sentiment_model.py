from modules import NeuralNetworkHiddenLayer
from data_batcher import SentimentDataObject
import tensorflow as tf
from tensorflow.python.ops import variable_scope as vs

class SentimentVanillaNeuralNetworkModel(object):
    def __init__(self,FLAGS,word2id,char2id,word_embed_matrix,char_embed_matrix):
        self.FLAGS=FLAGS
        self.dataObject=SentimentDataObject(word2id,char2id,word_embed_matrix,char_embed_matrix,self.FLAGS.train_path,self.FLAGS.test_path,self.FLAGS.batch_size,self.FLAGS.review_length,self.FLAGS.word_length,self.FLAGS.discard_long,self.FLAGS.test_size)

        with tf.variable_scope('SentimentModel',initializer=tf.contrib.layers.variance_scaling_initializer(factor=1.0,uniform=True)):
            self.add_placeholders()
            self.build_graph()
            self.add_loss()
            self.add_training_step()

        #params=tf.trainable_variables()
        #gradients=tf.gradients(self.loss,params)
        #self.global_step=tf.Variable(0,name="global_step",trainable=False)
        #opt=tf.train.AdamOptimizer(learning_rate=FLAGS.learning_rate)
        #self.train_step=tf.train.AdamOptimizer(learning_rate=FLAGS.learning_rate).minimize(self.loss)

    def add_placeholders(self):
        self.review_words=tf.placeholder(dtype=tf.float32,shape=[None,self.FLAGS.review_length,self.FLAGS.word_embedding_size])
        self.review_mask=tf.placeholder(dtype=tf.float32,shape=[None,self.FLAGS.review_length])
        self.char_words=tf.placeholder(dtype=tf.float32,shape=[None,self.FLAGS.review_length,self.FLAGS.word_length,self.FLAGS.char_embedding_size])
        self.ratings=tf.placeholder(dtype=tf.float32,shape=[None,self.FLAGS.ratings_size])
        self.keep_prob = tf.placeholder_with_default(1.0, shape=())

    def build_graph(self):
        #first collapse the review_words
        review_sum=tf.reduce_sum(self.review_words, 1)  # (batch,word_embedding_size)
        mask_sum=tf.reduce_sum(self.review_mask,1,keepdims=True)+tf.constant(0.0001, dtype=tf.float32,shape=())  # (batch,1)
        # bag of words
        input_vector=review_sum/mask_sum

        HIDDEN_LAYER1=2048
        HIDDEN_LAYER2=2048
        HIDDEN_LAYER3=2048
        layer1=NeuralNetworkHiddenLayer('HiddenLayer1',self.FLAGS.word_embedding_size,HIDDEN_LAYER1,self.keep_prob)
        output1=layer1.build_graph(input_vector)
        layer2=NeuralNetworkHiddenLayer('HiddenLayer2',self.HIDDEN_LAYER1,HIDDEN_LAYER2,self.keep_prob)
        output2=layer2.build_graph(output1)
        layer3=NeuralNetworkHiddenLayer('HiddenLayer3',self.HIDDEN_LAYER2, HIDDEN_LAYER3, self.keep_prob)
        output3=layer3.build_graph(output2)
        fully_connected_weight1=tf.get_variable(name='fully_connected1_weight',shape=[HIDDEN_LAYER3,self.FLAGS.ratings_size],dtype=tf.float32,initializer=tf.truncated_normal_initializer(stddev=0.05))
        fully_connected_bias=tf.get_variable(name='fully_connected1_bias',shape=[self.FLAGS.ratings_size],dtype=tf.float32,initializer=tf.constant_initializer(0.0))
        final_output=tf.matmul(output3,fully_connected_weight1)+fully_connected_bias
        #tf.identity used to explicitly transport tensor between devices
        logits=tf.identity(final_output, name="logits")
        self.final_output=final_output
        self.logits=logits

    def add_loss(self):
        with vs.variable_scope('loss'):
            self.loss=tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=self.logits, labels=self.ratings))
            tf.summary.scalar('loss',self.loss)

    def add_training_step(self):
        self.train_step=tf.train.AdamOptimizer(learning_rate=self.FLAGS.learning_rate).minimize(self.loss)

    def correct_pred(self):
        correct_prediction=tf.equal(tf.argmax(self.final_output, 1),tf.argmax(self.ratings, 1),name='correct_prediction')
        return correct_prediction

    def accuracy(self,correct_prediction):
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32), name='accuracy')
        return accuracy

    def run_train_iter(self,):



