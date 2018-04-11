#import sys
#sys.path.append('../python_libraries')
from tensorflow_modules.modules import NeuralNetworkHiddenLayer,Convolution1DLayer,LSTMLayer
from tensorflow_modules.model import Model
from data_batcher import SentimentDataObject
import tensorflow as tf
from tensorflow.python.ops import variable_scope as vs
from math import ceil

class SentimentVanillaNeuralNetworkModel(Model):
    def __init__(self,FLAGS,word2id,char2id,word_embed_matrix,char_embed_matrix):
        self.FLAGS=FLAGS
        self.dataObject=SentimentDataObject(word2id,char2id,word_embed_matrix,char_embed_matrix,self.FLAGS.train_path,self.FLAGS.test_path,self.FLAGS.batch_size,self.FLAGS.review_length,self.FLAGS.word_length,self.FLAGS.discard_long,self.FLAGS.test_size)

        with tf.variable_scope('SentimentModel',initializer=tf.contrib.layers.variance_scaling_initializer(factor=1.0,uniform=True)):
            self.add_placeholders()
            self.build_graph()
            self.add_loss()
            self.add_training_step()

    def add_placeholders(self):
        self.review_words=tf.placeholder(dtype=tf.float32,shape=[None,self.FLAGS.review_length,self.FLAGS.word_embedding_size])
        self.review_mask=tf.placeholder(dtype=tf.float32,shape=[None,self.FLAGS.review_length])
        self.char_words=tf.placeholder(dtype=tf.float32,shape=[None,self.FLAGS.review_length,self.FLAGS.word_length,self.FLAGS.char_embedding_size])
        self.ratings=tf.placeholder(dtype=tf.float32,shape=[None,self.FLAGS.ratings_size])
        self.keep_prob = tf.placeholder_with_default(1.0, shape=())

    def build_graph(self):
        #first collapse the review_words
        HIDDEN_LAYER1_SIZE=2048
        HIDDEN_LAYER2_SIZE=2048
        HIDDEN_LAYER3_SIZE=2048

        review_sum=tf.reduce_sum(self.review_words,1)  #(batch,word_embedding_size)
        mask_sum=tf.reduce_sum(self.review_mask,1,keepdims=True)+tf.constant(0.0001,dtype=tf.float32,shape=())  #(batch,1)
        # bag of words
        input_vector=review_sum/mask_sum    #(batch,word_embedding_size

        layer1=NeuralNetworkHiddenLayer('HiddenLayer1',self.FLAGS.word_embedding_size,HIDDEN_LAYER1_SIZE,self.keep_prob)
        output1=layer1.build_graph(input_vector)    #(batch,hidden_layer1_size)

        layer2=NeuralNetworkHiddenLayer('HiddenLayer2',HIDDEN_LAYER1_SIZE,HIDDEN_LAYER2_SIZE,self.keep_prob)
        output2=layer2.build_graph(output1)         #(batch,hidden_layer2_size)

        layer3=NeuralNetworkHiddenLayer('HiddenLayer3',HIDDEN_LAYER2_SIZE,HIDDEN_LAYER3_SIZE,self.keep_prob)
        output3=layer3.build_graph(output2)         #(batch,hidden_layer3_size)

        with tf.variable_scope('full_layer1') as scope:
            full_weight1=tf.get_variable(name='full_layer1_weight',shape=[HIDDEN_LAYER3_SIZE,self.FLAGS.ratings_size],dtype=tf.float32,initializer=tf.truncated_normal_initializer(stddev=0.05))
            full_bias1=tf.get_variable(name='full_layer_bias',shape=[self.FLAGS.ratings_size],dtype=tf.float32,initializer=tf.constant_initializer(0.0))
            final_output=tf.matmul(output3,full_weight1)+full_bias1

        logits = tf.identity(final_output, name="logits")

        self.final_output=final_output
        self.logits=logits

    def add_loss(self):
        with vs.variable_scope('loss'):
            self.cost=tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=self.logits, labels=self.ratings))
            self.correct_prediction = tf.equal(tf.argmax(self.final_output,1),tf.argmax(self.ratings,1),name='correct_prediction')
            self.accuracy=tf.reduce_mean(tf.cast(self.correct_prediction,tf.float32),name='accuracy')
            tf.summary.scalar('cost',self.cost)

    def add_training_step(self):
        self.train_step=tf.train.AdamOptimizer(learning_rate=self.FLAGS.learning_rate).minimize(self.cost)

    def run_train_iter(self,sess,input_batch,labels_batch):
        review_words_batch,review_chars_batch,review_mask_batch=input_batch
        train_data_feed = {
            self.review_words:review_words_batch,
            self.char_words:review_chars_batch,
            self.review_mask:review_mask_batch,
            self.ratings:labels_batch,
            self.keep_prob:(1.0-self.FLAGS.dropout),
        }
        sess.run(self.train_step,feed_dict=train_data_feed)

    def get_validation_accuracy(self,sess):
        validation_accuracy = 0.0
        num_batches = 0
        for review_words_batch, review_chars_batch, review_mask_batch, ratings_batch in self.dataObject.generate_dev_data():
            num_batches += 1
            dev_data_feed = {
                self.review_words: review_words_batch,
                self.char_words: review_chars_batch,
                self.review_mask: review_mask_batch,
                self.ratings: ratings_batch,
                self.keep_prob: 1.0,
            }
            validation_accuracy_batch = sess.run([self.accuracy], dev_data_feed)
            validation_accuracy += validation_accuracy_batch[0]
        validation_accuracy /= num_batches
        return validation_accuracy

    def get_test_data(self,sess):
        output = []
        lineids = []
        for review_words_batch, review_chars_batch, review_mask_batch, lineid_batch in self.dataObject.generate_test_data():
            test_data_feed = {
                self.review_words: review_words_batch,
                self.char_words: review_chars_batch,
                self.review_mask: review_mask_batch,
                self.keep_prob: 1.0,
            }
            test_output = sess.run(tf.argmax(self.final_output, 1), feed_dict=test_data_feed)
            lineids.extend(lineid_batch.tolist())
            output.extend(test_output.tolist())
        return lineids,output

    def run_epoch(self,sess):
        for review_words_batch, review_chars_batch, review_mask_batch, ratings_batch in self.dataObject.generate_one_epoch():
            input_batch=(review_words_batch, review_chars_batch, review_mask_batch)
            self.run_train_iter(sess,input_batch, ratings_batch)
        validation_accuracy=self.get_validation_accuracy(sess)
        return validation_accuracy

class SentimentLSTMNeuralNetwork(Model):
    def __init__(self,FLAGS,word2id,char2id,word_embed_matrix,char_embed_matrix):
        self.FLAGS=FLAGS
        self.dataObject=SentimentDataObject(word2id,char2id,word_embed_matrix,char_embed_matrix,self.FLAGS.train_path,self.FLAGS.test_path, self.FLAGS.batch_size,self.FLAGS.review_length,self.FLAGS.word_length,self.FLAGS.discard_long,self.FLAGS.test_size)

        with tf.variable_scope('SentimentModel',initializer=tf.contrib.layers.variance_scaling_initializer(factor=1.0, uniform=True)):
            self.add_placeholders()
            self.build_graph()
            self.add_loss()
            self.add_training_step()

    def add_placeholders(self):
        self.review_words=tf.placeholder(dtype=tf.float32,shape=[None,self.FLAGS.review_length,self.FLAGS.word_embedding_size])
        self.review_mask=tf.placeholder(dtype=tf.float32,shape=[None,self.FLAGS.review_length])
        self.char_words=tf.placeholder(dtype=tf.float32,shape=[None,self.FLAGS.review_length,self.FLAGS.word_length,self.FLAGS.char_embedding_size])
        self.ratings=tf.placeholder(dtype=tf.float32,shape=[None,self.FLAGS.ratings_size])
        self.keep_prob = tf.placeholder_with_default(1.0, shape=())

    def build_graph(self):
        hidden_size=300
        num_layers=1
        layer1=LSTMLayer(name='LSTMLayer1',hidden_size=hidden_size,keep_prop=self.keep_prob,num_layers=num_layers)
        output1=layer1.build_graph(self.review_words,self.review_mask)
        with tf.variable_scope('full_layer1') as scope:
            full_weight1=tf.get_variable(name='full_layer1_weight',shape=[hidden_size,self.FLAGS.ratings_size],dtype=tf.float32,initializer=tf.truncated_normal_initializer(stddev=0.05))
            full_bias1=tf.get_variable(name='full_layer_bias',shape=[self.FLAGS.ratings_size],dtype=tf.float32,initializer=tf.constant_initializer(0.0))
            final_output=tf.matmul(output1,full_weight1)+full_bias1
        logits = tf.identity(final_output, name="logits")
        self.final_output=final_output
        self.logits=logits

    def add_loss(self):
        with vs.variable_scope('loss'):
            self.cost=tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=self.logits, labels=self.ratings))
            self.correct_prediction = tf.equal(tf.argmax(self.final_output,1),tf.argmax(self.ratings,1),name='correct_prediction')
            self.accuracy=tf.reduce_mean(tf.cast(self.correct_prediction,tf.float32),name='accuracy')
            tf.summary.scalar('cost',self.cost)

    def add_training_step(self):
        self.train_step=tf.train.AdamOptimizer(learning_rate=self.FLAGS.learning_rate).minimize(self.cost)

    def run_train_iter(self,sess,input_batch,labels_batch):
        review_words_batch,review_chars_batch,review_mask_batch=input_batch
        train_data_feed = {
            self.review_words:review_words_batch,
            self.char_words:review_chars_batch,
            self.review_mask:review_mask_batch,
            self.ratings:labels_batch,
            self.keep_prob:(1.0-self.FLAGS.dropout),
        }
        sess.run(self.train_step,feed_dict=train_data_feed)

    def get_validation_accuracy(self,sess):
        validation_accuracy = 0.0
        num_batches = 0
        for review_words_batch, review_chars_batch, review_mask_batch, ratings_batch in self.dataObject.generate_dev_data():
            num_batches += 1
            dev_data_feed = {
                self.review_words: review_words_batch,
                self.char_words: review_chars_batch,
                self.review_mask: review_mask_batch,
                self.ratings: ratings_batch,
                self.keep_prob: 1.0,
            }
            validation_accuracy_batch = sess.run([self.accuracy], dev_data_feed)
            validation_accuracy += validation_accuracy_batch[0]
        validation_accuracy /= num_batches
        return validation_accuracy

    def get_test_data(self,sess):
        output = []
        lineids = []
        for review_words_batch, review_chars_batch, review_mask_batch, lineid_batch in self.dataObject.generate_test_data():
            test_data_feed = {
                self.review_words: review_words_batch,
                self.char_words: review_chars_batch,
                self.review_mask: review_mask_batch,
                self.keep_prob: 1.0,
            }
            test_output = sess.run(tf.argmax(self.final_output, 1), feed_dict=test_data_feed)
            lineids.extend(lineid_batch.tolist())
            output.extend(test_output.tolist())
        return lineids,output

    def run_epoch(self,sess):
        for review_words_batch, review_chars_batch, review_mask_batch, ratings_batch in self.dataObject.generate_one_epoch():
            input_batch=(review_words_batch, review_chars_batch, review_mask_batch)
            self.run_train_iter(sess,input_batch, ratings_batch)
        validation_accuracy=self.get_validation_accuracy(sess)
        return validation_accuracy

class SentimentWordCNNNeuralNetwork(Model):
    def __init__(self,FLAGS,word2id,char2id,word_embed_matrix,char_embed_matrix):
        self.FLAGS=FLAGS
        self.dataObject=SentimentDataObject(word2id,char2id,word_embed_matrix,char_embed_matrix,self.FLAGS.train_path,self.FLAGS.test_path,self.FLAGS.batch_size,self.FLAGS.review_length,self.FLAGS.word_length,self.FLAGS.discard_long,self.FLAGS.test_size)

        with tf.variable_scope('SentimentModel',initializer=tf.contrib.layers.variance_scaling_initializer(factor=1.0,uniform=True)):
            self.add_placeholders()
            self.build_graph()
            self.add_loss()
            self.add_training_step()

    def add_placeholders(self):
        self.review_words=tf.placeholder(dtype=tf.float32,shape=[None,self.FLAGS.review_length,self.FLAGS.word_embedding_size])
        self.review_mask=tf.placeholder(dtype=tf.float32,shape=[None,self.FLAGS.review_length])
        self.char_words=tf.placeholder(dtype=tf.float32,shape=[None,self.FLAGS.review_length,self.FLAGS.word_length,self.FLAGS.char_embedding_size])
        self.ratings=tf.placeholder(dtype=tf.float32,shape=[None,self.FLAGS.ratings_size])
        self.keep_prob = tf.placeholder_with_default(1.0, shape=())

    def build_graph(self):
        channel_size_1=600
        filter1=[3,self.FLAGS.word_embedding_size,channel_size_1]
        strides_1=1
        pool_size_1=[2]
        layer1=Convolution1DLayer('ConvLayer1',filter1,strides_1,channel_size_1,pool_size_1)
        output1=layer1.build_graph(self.review_words)
        output1_shape=[1,ceil(ceil(self.FLAGS.review_length/strides_1)/pool_size_1[0]),channel_size_1]

        channel_size_2=600
        filter2=[3,channel_size_1,channel_size_2]
        strides_2=1
        pool_size_2=[2]
        layer2=Convolution1DLayer('ConvLayer2',filter2,strides_2,channel_size_2,pool_size_2)
        output2=layer2.build_graph(output1)
        output2_shape=[1,ceil(ceil(output1_shape[1]/strides_2)/pool_size_2[0]), channel_size_2]

        channel_size_3=300
        filter3=[3,channel_size_2,channel_size_3]
        strides_3=1
        pool_size_3=[2]
        layer3=Convolution1DLayer('ConvLayer3',filter3,strides_3,channel_size_3,pool_size_3)
        output3=layer3.build_graph(output2)
        output3_shape=[1,ceil(ceil(output2_shape[1]/strides_3)/pool_size_3[0]),channel_size_3]

        reshaped_output=tf.reshape(output3,[-1,output3_shape[1]*output3_shape[2]])
        HIDDEN_LAYER1_SIZE=2048
        HIDDEN_LAYER2_SIZE=1024
        HIDDEN_LAYER3_SIZE=512
        layer4=NeuralNetworkHiddenLayer('HiddenLayer1',output3_shape[1]*output3_shape[2],HIDDEN_LAYER1_SIZE,self.keep_prob)
        output4=layer4.build_graph(reshaped_output)

        layer5=NeuralNetworkHiddenLayer('HiddenLayer2',HIDDEN_LAYER1_SIZE,HIDDEN_LAYER2_SIZE,self.keep_prob)
        output5=layer5.build_graph(output4)

        layer6=NeuralNetworkHiddenLayer('HiddenLayer3',HIDDEN_LAYER2_SIZE,HIDDEN_LAYER3_SIZE,self.keep_prob)
        output6=layer6.build_graph(output5)  # (batch,hidden_layer3_size)

        with tf.variable_scope('full_layer1') as scope:
            full_weight1 = tf.get_variable(name='full_layer1_weight',shape=[HIDDEN_LAYER3_SIZE,self.FLAGS.ratings_size],dtype=tf.float32,initializer=tf.truncated_normal_initializer(stddev=0.05))
            full_bias1 = tf.get_variable(name='full_layer_bias', shape=[self.FLAGS.ratings_size], dtype=tf.float32,initializer=tf.constant_initializer(0.0))
            final_output = tf.matmul(output6, full_weight1)+full_bias1

        logits = tf.identity(final_output, name="logits")
        self.final_output = final_output
        self.logits = logits

    def add_loss(self):
        with vs.variable_scope('loss'):
            self.cost=tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=self.logits, labels=self.ratings))
            self.correct_prediction = tf.equal(tf.argmax(self.final_output,1),tf.argmax(self.ratings,1),name='correct_prediction')
            self.accuracy=tf.reduce_mean(tf.cast(self.correct_prediction,tf.float32),name='accuracy')
            tf.summary.scalar('cost',self.cost)

    def add_training_step(self):
        self.train_step=tf.train.AdamOptimizer(learning_rate=self.FLAGS.learning_rate).minimize(self.cost)

    def run_train_iter(self,sess,input_batch,labels_batch):
        review_words_batch,review_chars_batch,review_mask_batch=input_batch
        train_data_feed = {
            self.review_words:review_words_batch,
            self.char_words:review_chars_batch,
            self.review_mask:review_mask_batch,
            self.ratings:labels_batch,
            self.keep_prob:(1.0-self.FLAGS.dropout),
        }
        sess.run(self.train_step,feed_dict=train_data_feed)

    def get_validation_accuracy(self,sess):
        validation_accuracy = 0.0
        num_batches = 0
        for review_words_batch, review_chars_batch, review_mask_batch, ratings_batch in self.dataObject.generate_dev_data():
            num_batches += 1
            dev_data_feed = {
                self.review_words: review_words_batch,
                self.char_words: review_chars_batch,
                self.review_mask: review_mask_batch,
                self.ratings: ratings_batch,
                self.keep_prob: 1.0,
            }
            validation_accuracy_batch = sess.run([self.accuracy], dev_data_feed)
            validation_accuracy += validation_accuracy_batch[0]
        validation_accuracy /= num_batches
        return validation_accuracy

    def get_test_data(self,sess):
        output = []
        lineids = []
        for review_words_batch, review_chars_batch, review_mask_batch, lineid_batch in self.dataObject.generate_test_data():
            test_data_feed = {
                self.review_words: review_words_batch,
                self.char_words: review_chars_batch,
                self.review_mask: review_mask_batch,
                self.keep_prob: 1.0,
            }
            test_output = sess.run(tf.argmax(self.final_output, 1), feed_dict=test_data_feed)
            lineids.extend(lineid_batch.tolist())
            output.extend(test_output.tolist())
        return lineids,output

    def run_epoch(self,sess):
        for review_words_batch, review_chars_batch, review_mask_batch, ratings_batch in self.dataObject.generate_one_epoch():
            input_batch=(review_words_batch, review_chars_batch, review_mask_batch)
            self.run_train_iter(sess,input_batch, ratings_batch)
        validation_accuracy=self.get_validation_accuracy(sess)
        return validation_accuracy

