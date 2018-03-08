import tensorflow as tf
import SentimentData
#import numpy as np
import pandas as pd

sentimentData=SentimentData.SentimentDataObject()

KEEP_PROB=0.5
INPUT_VECTOR_SIZE=sentimentData.X_train.shape[1]
HIDDEN_LAYER1_SIZE=2048
HIDDEN_LAYER2_SIZE=2048
HIDDEN_LAYER3_SIZE=2048
OUTPUT_SIZE=sentimentData.y_train.shape[1]
LEARNING_RATE=0.001
NUM_EPOCHS=100
BATCH_SIZE=10000

def truncated_normal_var(name, shape, dtype):
    return (tf.get_variable(name=name, shape=shape, dtype=dtype, initializer=tf.truncated_normal_initializer(stddev=0.05)))

def zero_var(name, shape, dtype):
    return (tf.get_variable(name=name, shape=shape, dtype=dtype, initializer=tf.constant_initializer(0.0)))

X=tf.placeholder(tf.float32,shape=[None,INPUT_VECTOR_SIZE],name='X')
labels=tf.placeholder(tf.float32,shape=[None,OUTPUT_SIZE],name='labels')

with tf.variable_scope('hidden_layer1') as scope:
    hidden_weight1=truncated_normal_var(name='hidden_weight1',shape=[INPUT_VECTOR_SIZE,HIDDEN_LAYER1_SIZE],dtype=tf.float32)
    hidden_bias1=zero_var(name='hidden_bias1',shape=[HIDDEN_LAYER1_SIZE],dtype=tf.float32)
    hidden_layer1=tf.nn.relu(tf.matmul(X,hidden_weight1)+hidden_bias1)
    hidden_layer1_drop=tf.nn.dropout(hidden_layer1,KEEP_PROB)

with tf.variable_scope('hidden_layer2') as scope:
    hidden_weight2=truncated_normal_var(name='hidden_weight2',shape=[HIDDEN_LAYER1_SIZE,HIDDEN_LAYER2_SIZE],dtype=tf.float32)
    hidden_bias2=zero_var(name='hidden_bias2',shape=[HIDDEN_LAYER2_SIZE],dtype=tf.float32)
    hidden_layer2=tf.nn.relu(tf.matmul(hidden_layer1_drop,hidden_weight2)+hidden_bias2)
    hidden_layer2_drop=tf.nn.dropout(hidden_layer2,KEEP_PROB)

with tf.variable_scope('hidden_layer3') as scope:
    hidden_weight3=truncated_normal_var(name='hidden_weight3',shape=[HIDDEN_LAYER2_SIZE,HIDDEN_LAYER3_SIZE],dtype=tf.float32)
    hidden_bias3=zero_var(name='hidden_bias3',shape=[HIDDEN_LAYER2_SIZE],dtype=tf.float32)
    hidden_layer3=tf.nn.relu(tf.matmul(hidden_layer2_drop,hidden_weight3)+hidden_bias3)
    hidden_layer3_drop=tf.nn.dropout(hidden_layer3,KEEP_PROB)

with tf.variable_scope('full_layer') as scope:
    full_weight1=truncated_normal_var(name='full_weight1',shape=[HIDDEN_LAYER3_SIZE,OUTPUT_SIZE],dtype=tf.float32)
    full_bias1 = zero_var(name='full_bias1', shape=[OUTPUT_SIZE], dtype=tf.float32)
    final_output=tf.matmul(hidden_layer3_drop,full_weight1)+full_bias1

logits=tf.identity(final_output,name="logits")

cost=tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits,labels=labels))
train_step=tf.train.AdamOptimizer(learning_rate=LEARNING_RATE).minimize(cost)
correct_prediction=tf.equal(tf.argmax(final_output,1),tf.argmax(labels,1),name='correct_prediction')
accuracy=tf.reduce_mean(tf.cast(correct_prediction,tf.float32),name='accuracy')

init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)

test_data_feed = {
    X: sentimentData.X_cv,
    labels: sentimentData.y_cv,
}

for epoch in range(NUM_EPOCHS):
    for batch_X, batch_y in sentimentData.generate_one_epoch_for_neural(BATCH_SIZE):
        train_data_feed = {
            X: batch_X,
            labels: batch_y,
        }
        sess.run(train_step, feed_dict={X:batch_X,labels:batch_y,})
    validation_accuracy=sess.run([accuracy], test_data_feed)
    print('validation_accuracy => '+str(validation_accuracy))

validation_accuracy=sess.run([accuracy], test_data_feed)
print('Final validation_accuracy => ' +str(validation_accuracy))

#generate the submission file
num_batches=int(sentimentData.test_data.shape[0])//BATCH_SIZE
if BATCH_SIZE*num_batches<sentimentData.test_data.shape[0]:
    num_batches+=1
output=[]
for j in range(num_batches):
    batch_X=sentimentData.test_data[j*BATCH_SIZE:(j + 1)*BATCH_SIZE]
    test_output=sess.run(tf.argmax(final_output,1),feed_dict={X:batch_X})
    output.extend(test_output.tolist())
    #print(len(output))


sentimentData.df_test_input['Classification']=pd.Series(output)
#print(sentimentData.df_test_input.head())
#sentimentData.df_test_input['Sentiment']=sentimentData.df_test_input['Classification'].apply(lambda x:sentimentData.lb.inverse_transform(x))
sentimentData.df_test_input['Sentiment']=sentimentData.df_test_input['Classification'].apply(lambda x:x)
#print(sentimentData.df_test_input.head())
submission=sentimentData.df_test_input[['PhraseId','Sentiment']]
submission.to_csv('submission.csv',index=False)