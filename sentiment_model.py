from modules import NeuralNetworkHiddenLayer
from data_batcher import SentimentDataObject
import tensorflow as tf
from tensorflow.python.ops import variable_scope as vs
from model import Model

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
        input_vector=review_sum/mask_sum

        layer1=NeuralNetworkHiddenLayer('HiddenLayer1',self.FLAGS.word_embedding_size,HIDDEN_LAYER1_SIZE,self.keep_prob)
        output1=layer1.build_graph(input_vector)

        layer2=NeuralNetworkHiddenLayer('HiddenLayer2',HIDDEN_LAYER1_SIZE,HIDDEN_LAYER2_SIZE,self.keep_prob)
        output2=layer2.build_graph(output1)

        layer3=NeuralNetworkHiddenLayer('HiddenLayer3',HIDDEN_LAYER2_SIZE,HIDDEN_LAYER3_SIZE,self.keep_prob)
        output3=layer3.build_graph(output2)

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
        print('validation_accuracy => ' + str(validation_accuracy))
        return validation_accuracy





