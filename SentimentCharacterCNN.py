import tensorflow as tf
import SentimentCharacterCNNData

sentimentCharacterData=SentimentCharacterCNNData.SentimentCharacterCNNObject()

def truncated_normal_var(name, shape, dtype):
  return(tf.get_variable(name=name, shape=shape, dtype=dtype, initializer=tf.truncated_normal_initializer(stddev=0.05)))
def zero_var(name, shape, dtype):
  return(tf.get_variable(name=name, shape=shape, dtype=dtype, initializer=tf.constant_initializer(0.0)))

KEEP_PROB=0.5
CONV_SHAPE=64
LEARNING_RATE=0.001
NUM_EPOCHS=100
BATCH_SIZE=10000

x=tf.placeholder(tf.float32,shape=[None,sentimentCharacterData.X_train.shape[1],sentimentCharacterData.X_train.shape[2]],name='x')
labels=tf.placeholder(tf.float32,shape=[None,sentimentCharacterData.y_train.shape[1]],name='labels')
keep_prob=tf.placeholder(tf.float32,name='keep_prob')

with tf.variable_scope('conv1') as scope:
    conv1_filter=truncated_normal_var(name='conv1_filter',shape=[3,sentimentCharacterData.X_train.shape[2],CONV_SHAPE],dtype=tf.float32)
    stride=1
    conv1=tf.nn.conv1d(x,conv1_filter,stride,padding='SAME')
    conv1_bias=zero_var(name='conv1_bias', shape=[CONV_SHAPE], dtype=tf.float32)
    conv1_add_bias=tf.nn.bias_add(conv1, conv1_bias)
    relu_conv1=tf.nn.relu(conv1_add_bias)

pool_size=[3]
pool1=tf.nn.pool(relu_conv1,pool_size,'MAX','SAME',strides=pool_size)

with tf.variable_scope('conv2') as scope:
    conv2_filter=truncated_normal_var(name='conv2_filter',shape=[3,CONV_SHAPE,CONV_SHAPE],dtype=tf.float32)
    stride=1
    conv2=tf.nn.conv1d(pool1,conv2_filter,stride,padding='SAME')
    conv2_bias=zero_var(name='conv2_bias',shape=[CONV_SHAPE],dtype=tf.float32)
    conv2_add_bias=tf.nn.bias_add(conv2, conv2_bias)
    relu_conv2=tf.nn.relu(conv2_add_bias)

pool_size=[3]
pool2=tf.nn.pool(relu_conv2,pool_size,'MAX','SAME',strides=pool_size)

zz=pool2.get_shape()
print(zz)
reshaped_output=tf.layers.flatten(pool2,'flattened_output')
reshaped_dim=reshaped_output.get_shape()[1].value
with tf.variable_scope('full1') as scope:
  full_weight1=truncated_normal_var(name='full_mult1',shape=[reshaped_dim,1024],dtype=tf.float32)
  full_bias1=zero_var(name='full_bias1',shape=[1024],dtype=tf.float32)
  full_layer1=tf.nn.relu(tf.add(tf.matmul(reshaped_output,full_weight1),full_bias1))
  full_layer1=tf.nn.dropout(full_layer1,keep_prob)

with tf.variable_scope('full2') as scope:
  full_weight2=truncated_normal_var(name='full_mult2',shape=[1024, 256],dtype=tf.float32)
  full_bias2=zero_var(name='full_bias2',shape=[256],dtype=tf.float32)
  full_layer2=tf.nn.relu(tf.add(tf.matmul(full_layer1,full_weight2),full_bias2))
  full_layer2=tf.nn.dropout(full_layer2,keep_prob)

with tf.variable_scope('full3') as scope:
  full_weight3=truncated_normal_var(name='full_mult3',shape=[256,sentimentCharacterData.y_train[0].shape[0]],dtype=tf.float32)
  full_bias3=zero_var(name='full_bias3',shape=[sentimentCharacterData.y_train[0].shape[0]],dtype=tf.float32)
  final_output=tf.add(tf.matmul(full_layer2,full_weight3),full_bias3,name='final_output')

logits=tf.identity(final_output,name='logits')
cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits,labels=labels),name='cross_entropy')
train_step=tf.train.AdamOptimizer(LEARNING_RATE).minimize(cross_entropy)
correct_prediction=tf.equal(tf.argmax(final_output,1),tf.argmax(labels,1),name='correct_prediction')
accuracy=tf.reduce_mean(tf.cast(correct_prediction,tf.float32),name='accuracy')

init=tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)
test_data_feed = {
    x: sentimentCharacterData.X_cv,
    labels: sentimentCharacterData.y_cv,
    keep_prob: 1.0,
}

for batch_X,batch_y in sentimentCharacterData.generate_one_epoch(BATCH_SIZE):
    train_data_feed = {
        x:batch_X,
        labels:batch_y,
        keep_prob:KEEP_PROB,
    }
    sess.run(train_step, feed_dict=train_data_feed)
    validation_accuracy = sess.run([accuracy], test_data_feed)
    print('validation_accuracy => ' + str(validation_accuracy))

validation_accuracy=sess.run([accuracy], test_data_feed)
print('Final validation_accuracy => ' +str(validation_accuracy))