from word_and_character_vectors import get_char,get_glove
import tensorflow as tf
import os
from datetime import datetime
from sentiment_model import SentimentVanillaNeuralNetworkModel

tf.app.flags.DEFINE_integer("gpu", 1, "Which GPU to use, if you have multiple.")
tf.app.flags.DEFINE_integer("num_epochs",100, "Number of epochs to train. 0 means train indefinitely")

# Hyperparameters
tf.app.flags.DEFINE_float("learning_rate",0.001,"Learning rate.")
tf.app.flags.DEFINE_float("dropout",0.5,"Fraction of units randomly dropped on non-recurrent connections.")
tf.app.flags.DEFINE_integer("batch_size",20000,"Batch size to use")
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

emb_matrix_char, char2id, id2char=get_char('../ml_data_files')
emb_matrix_word, word2id, id2word=get_glove('../ml_data_files')

sentiment_model=SentimentVanillaNeuralNetworkModel(FLAGS,word2id,char2id,emb_matrix_word,emb_matrix_char)

init=tf.global_variables_initializer()
sess=tf.Session()
sess.run(init)
print(FLAGS.num_epochs)
for epoch in range(FLAGS.num_epochs):
    validation_accuracy=sentiment_model.run_epoch(sess)

print('Final validation_accuracy => ' +str(sentiment_model.get_validation_accuracy(sess)))

lineids,output=sentiment_model.get_test_data(sess)
filename='submission_'+datetime.now().strftime('%Y%m%d%I%M')+'.csv'
with open(filename,'w') as f:
    f.write("PhraseId,Sentiment\n")
    for i,item in enumerate(output):
        f.write(str(lineids[i])+','+str(item)+'\n')
