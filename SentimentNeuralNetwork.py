from word_and_character_vectors import get_char,get_glove
import tensorflow as tf
from data_batcher import SentimentDataObject
import os
from modules import NeuralNetworkHiddenLayer

tf.app.flags.DEFINE_integer("gpu", 1, "Which GPU to use, if you have multiple.")
tf.app.flags.DEFINE_integer("num_epochs",2, "Number of epochs to train. 0 means train indefinitely")

# Hyperparameters
tf.app.flags.DEFINE_float("learning_rate",0.001,"Learning rate.")
tf.app.flags.DEFINE_float("dropout",0.5,"Fraction of units randomly dropped on non-recurrent connections.")
tf.app.flags.DEFINE_integer("batch_size",10000,"Batch size to use")
###tf.app.flags.DEFINE_integer("hidden_size",200,"Size of the hidden states")
tf.app.flags.DEFINE_integer("review_length",57,"The maximum words in each review")
tf.app.flags.DEFINE_integer("word_length", 15, "The maximum characters in each word")
tf.app.flags.DEFINE_integer("word_embedding_size", 300, "Size of the pretrained word vectors.")
tf.app.flags.DEFINE_integer("char_embedding_size", 128, "Size of the pretrained char vectors.")
tf.app.flags.DEFINE_float("test_size",0.10,"Dev set to split from training set")
tf.app.flags.DEFINE_string("train_path","data/train.tsv",'Location of training data')
tf.app.flags.DEFINE_string("test_path","data/test.tsv",'Location of training data')
tf.app.flags.DEFINE_bool("discard_long",False,"Discard lines longer than review_length")
tf.app.flags.DEFINE_integer("ratings_size",5, "Number of ratings.")

FLAGS = tf.app.flags.FLAGS
os.environ["CUDA_VISIBLE_DEVICES"] = str(FLAGS.gpu)

emb_matrix_char, char2id, id2char=get_char('C:\\Users\\tihor\\Documents\\ml_data_files')
emb_matrix_word, word2id, id2word=get_glove('C:\\Users\\tihor\\Documents\\ml_data_files')
dataObject=SentimentDataObject(word2id,char2id,emb_matrix_word,emb_matrix_char,FLAGS.train_path,FLAGS.test_path,FLAGS.batch_size,FLAGS.review_length,FLAGS.word_length,FLAGS.discard_long,FLAGS.test_size)

review_words=tf.placeholder(dtype=tf.float32,shape=[None,FLAGS.review_length,FLAGS.word_embedding_size])
review_mask=tf.placeholder(dtype=tf.float32,shape=[None,FLAGS.review_length])
char_words=tf.placeholder(dtype=tf.float32,shape=[None,FLAGS.review_length,FLAGS.word_length,FLAGS.char_embedding_size])
ratings=tf.placeholder(dtype=tf.float32,shape=[None,FLAGS.ratings_size])
keep_prob = tf.placeholder_with_default(1.0, shape=())

HIDDEN_LAYER1_SIZE=2048
HIDDEN_LAYER2_SIZE=2048
HIDDEN_LAYER3_SIZE=2048

review_sum=tf.reduce_sum(review_words,1)       #(batch,word_embedding_size)
mask_sum=tf.reduce_sum(review_mask,1,keepdims=True)+tf.constant(0.0001,dtype=tf.float32,shape=())   #(batch,1)
input_vector=review_sum/mask_sum

layer1=NeuralNetworkHiddenLayer('HiddenLayer1',FLAGS.word_embedding_size,HIDDEN_LAYER1_SIZE,keep_prob)
output1=layer1.build_graph(input_vector)

layer2=NeuralNetworkHiddenLayer('HiddenLayer2',HIDDEN_LAYER1_SIZE,HIDDEN_LAYER2_SIZE,keep_prob)
output2=layer2.build_graph(output1)

layer3=NeuralNetworkHiddenLayer('HiddenLayer3',HIDDEN_LAYER2_SIZE,HIDDEN_LAYER3_SIZE,keep_prob)
output3=layer3.build_graph(output2)

with tf.variable_scope('full_layer1') as scope:
    full_weight1=tf.get_variable(name='full_layer1_weight',shape=[HIDDEN_LAYER3_SIZE,FLAGS.ratings_size],dtype=tf.float32,initializer=tf.truncated_normal_initializer(stddev=0.05))
    full_bias1=tf.get_variable(name='full_layer_bias',shape=[FLAGS.ratings_size],dtype=tf.float32,initializer=tf.constant_initializer(0.0))
    final_output=tf.matmul(output3,full_weight1)+full_bias1

logits=tf.identity(final_output,name="logits")

cost=tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits,labels=ratings))
train_step=tf.train.AdamOptimizer(learning_rate=FLAGS.learning_rate).minimize(cost)
correct_prediction=tf.equal(tf.argmax(final_output,1),tf.argmax(ratings,1),name='correct_prediction')
accuracy=tf.reduce_mean(tf.cast(correct_prediction,tf.float32),name='accuracy')

init=tf.global_variables_initializer()
sess=tf.Session()
sess.run(init)

def get_validation_accuracy():
    validation_accuracy = 0.0
    num_batches = 0
    for review_words_batch, review_chars_batch, review_mask_batch, ratings_batch in dataObject.generate_dev_data():
        num_batches += 1
        dev_data_feed = {
            review_words: review_words_batch,
            char_words: review_chars_batch,
            review_mask: review_mask_batch,
            ratings: ratings_batch,
            keep_prob: 1.0,
        }
        validation_accuracy_batch = sess.run([accuracy], dev_data_feed)
        validation_accuracy += validation_accuracy_batch[0]
    validation_accuracy /= num_batches
    return validation_accuracy

for epoch in range(FLAGS.num_epochs):
    for review_words_batch,review_chars_batch,review_mask_batch,ratings_batch in dataObject.generate_one_epoch():
        train_data_feed={
            review_words: review_words_batch,
            char_words: review_chars_batch,
            review_mask: review_mask_batch,
            ratings: ratings_batch,
            keep_prob: (1.0-FLAGS.dropout),
        }
        sess.run(train_step, feed_dict=train_data_feed)
    print('validation_accuracy => ' + str(get_validation_accuracy()))

print('Final validation_accuracy => ' +str(get_validation_accuracy()))

