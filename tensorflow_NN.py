import tensorflow as tf

from tensorflow.examples.tutorials.mnist import input_data

def main():
	mnist = input_data.read_data_sets("/tmp/data",one_hot = True)
	hl1 = 500
	hl2 = 500
	hl3 = 500
	classes = 10
	batch_size = 100

	x = tf.placeholder('float',[None,784])
	y = tf.placeholder('float')

	hidden_layer_1 = {
		'weights': tf.Variable(tf.random_normal([784,hl1])),
		'biases' : tf.Variable(tf.random_normal([hl1]))
	}

	hidden_layer_2 = {
		'weights': tf.Variable(tf.random_normal([hl1,hl2])),
		'biases' : tf.Variable(tf.random_normal([hl2]))
	}

	hidden_layer_3 = {
		'weights': tf.Variable(tf.random_normal([hl2,hl3])),
		'biases' : tf.Variable(tf.random_normal([hl3]))
	}

	output_layer = {
		'weights': tf.Variable(tf.random_normal([hl3,classes])),
		'biases' : tf.Variable(tf.random_normal([classes]))
	}

	l1 = tf.add(tf.matmul(x,hidden_layer_1['weights']),hidden_layer_1['biases'])
	l1 = tf.nn.relu(l1)

	l2 = tf.add(tf.matmul(l1,hidden_layer_2['weights']),hidden_layer_2['biases'])
	l2 = tf.nn.relu(l2)

	l3 = tf.add(tf.matmul(l2,hidden_layer_3['weights']),hidden_layer_3['biases'])
	l3 = tf.nn.relu(l3)

	output = tf.matmul(l3,output_layer['weights'])+ output_layer['biases']

	cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=output,labels=y))
	optimizer = tf.train.AdamOptimizer().minimize(cost)

	epochs = 10
	with tf.Session() as sess:
		sess.run(tf.global_variables_initializer())
		for epoch in range(epochs):
			loss = 0
			for _ in range(int(mnist.train.num_examples/batch_size)):
				batch_x,batch_y = mnist.train.next_batch(batch_size)
				_, c = sess.run([optimizer,cost],feed_dict ={x:batch_x,y:batch_y})
				loss += c
			print("Loss of epochs{} = {}".format(epoch,loss))
		correct = tf.equal(tf.argmax(output, 1), tf.argmax(y, 1))
		accuracy = tf.reduce_mean(tf.cast(correct, 'float'))
		print("Accuracy = {}".format(accuracy.eval({x: mnist.test.images, y: mnist.test.labels})))

if __name__ == '__main__':
	main()