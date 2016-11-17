import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("/tmp/data/", one_hot = True)

n_classes = 10
batch_size = 128

x = tf.placeholder('float', [None, 784])
y = tf.placeholder('float')

#dropout settings
keep_rate = 0.8 #80% of the neurons are kept..
keep_prob = tf.placeholder(tf.float32)

def conv2d(x,W): #this is a sample meth for conv
    return tf.nn.conv2d(x,W,strides=[1,1,1,1],padding='SAME')

def maxpool(x): #maxpooling wrapper
    #                      size of window     moving strides
    return tf.nn.max_pool(x,ksize=[1,2,2,1],strides=[1,2,2,1],padding='SAME')


def cnn(x):
    #start with 28x28 image
    weights = {'W_conv1':tf.Variable(tf.random_normal([5,5,1,32])), #5x5 convulution, 1 input, 32 output/filters
                'W_conv2':tf.Variable(tf.random_normal([5,5,32,64])), #64 filters
                'W_fc':tf.Variable(tf.random_normal([7*7*64,1024])), #fully connected., image shrinked to 7 by 7, 1024 nodes in the layer
                'out':tf.Variable(tf.random_normal([1024,n_classes]))}

    biases = {'b_conv1':tf.Variable(tf.random_normal([32])), #5x5 convulution, 1 input, 32 output/filters
                'b_conv2':tf.Variable(tf.random_normal([64])), #64 filters
                'b_fc':tf.Variable(tf.random_normal([1024])), #fully connected., image shrinked to 7 by 7, 1024 nodes in the layer
                'out':tf.Variable(tf.random_normal([n_classes]))}
    
    #convert it to 2d matrix by reshape as conv, maxpool operate on 2d matrix 
    x = tf.reshape(x, shape=[-1,28,28,1])

    conv1 = conv2d(x,weights['W_conv1'])
    pool1 = maxpool(conv1)

    conv2 = conv2d(pool1,weights['W_conv2'])
    pool2 = maxpool(conv2)

    #since we are going to feed the image to fc NN, reshape to one dimention matrix
    fc = tf.reshape(pool2,shape=[-1,7*7*64])
    fc = tf.nn.relu(tf.matmul(fc,weights['W_fc'])+biases['b_fc'])
    
    #dropout -useful in very large dataset
    #fc = tf.nn.dropout(fc,keep_rate)

    output = tf.matmul(fc,weights['out']) + biases['out']

    return output

def train_neural_network(x):
    prediction = cnn(x)
    cost = tf.reduce_mean( tf.nn.softmax_cross_entropy_with_logits(prediction,y) )
    optimizer = tf.train.AdamOptimizer().minimize(cost)
    
    hm_epochs = 10
    with tf.Session() as sess:
        sess.run(tf.initialize_all_variables())

        for epoch in range(hm_epochs):
            epoch_loss = 0
            for _ in range(int(mnist.train.num_examples/batch_size)):
                epoch_x, epoch_y = mnist.train.next_batch(batch_size)
                _, c = sess.run([optimizer, cost], feed_dict={x: epoch_x, y: epoch_y})
                epoch_loss += c

            print('Epoch', epoch, 'completed out of',hm_epochs,'loss:',epoch_loss)

        correct = tf.equal(tf.argmax(prediction, 1), tf.argmax(y, 1))

        accuracy = tf.reduce_mean(tf.cast(correct, 'float'))
        print('Accuracy:',accuracy.eval({x:mnist.test.images, y:mnist.test.labels}))

train_neural_network(x)
