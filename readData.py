import scipy.io as sio
import  tensorflow as tf
import numpy as np
import random
import  matplotlib.pyplot as plt

data = sio.loadmat("/home/lijq/IdeaProjects/classAreticle/feature.mat")
#getdata and processdata
images_r = data['train_feature']
labels_rr =data['train_labels']
images_test = data['test_feature']
labels_test = data['test_label']
#
file = open('/home/lijq/IdeaProjects/classAreticle/resize/Testing/00000/00017_00000.png')
picdata = file.read()
file.close()

#
pic = tf.image.decode_png(picdata, channels=4)
pic = tf.expand_dims(pic, 0)

labels_r = np.array(labels_rr).transpose()
labels_testt = np.array(labels_test).transpose()
labels_input = []
labels_input_test = []
for i in range(len(labels_r)):
    labels_input.append(labels_r[i][0])
for j in range(len(labels_testt)):
    labels_input_test.append(labels_testt[j][0])

#build graph
with tf.name_scope(name='inputs_content'):
    x = tf.placeholder(dtype=tf.float32, shape=[None, 32, 32,3],name='my_input')
    y = tf.placeholder(dtype=tf.int32,shape=[None],name = 'my_label')
input_x = tf.contrib.layers.flatten(x)

# Fully connected layer
logits = tf.contrib.layers.fully_connected(input_x, 62, tf.nn.relu)
predicted_labels = tf.argmax(logits,1)

print np.shape(y),np.shape(logits)
# Define a loss function
loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels = y,
                                                                     logits = logits),name='loss_train')

# Define an optimizer
train_op = tf.train.AdamOptimizer(learning_rate=0.001).minimize(loss)

# Convert logits to label indexes
correct_pred = tf.argmax(logits, 1)
c_p = tf.equal(correct_pred,tf.cast(y,tf.int64))
# Define an accuracy metric
accuracy = tf.reduce_mean(tf.cast(c_p, tf.float32))
tf.summary.scalar('accuracy',accuracy)
tf.summary.scalar('loss',loss)
tf.summary.image('pic',pic)
print("images_flat: ", input_x)

print("logits: ", logits)

print("loss: ", loss)

print("predicted_labels: ", correct_pred)
print("accuracy: ", accuracy)

tf.set_random_seed(1234)
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    merged = tf.summary.merge_all()
    writer = tf.summary.FileWriter('./logs', sess.graph)
    for i in range(20):
        _, loss_value,accuracys,mer= sess.run([train_op,loss,accuracy,merged],
                                 feed_dict={x: images_r, y: labels_input})
        writer.add_summary(mer,global_step=i)
        if i % 10 == 0:
            print("Loss: ", loss_value,'ac:',accuracys)

    # Pick 10 random images
    sample_indexes = random.sample(range(len(images_r)), 10)
    sample_images = [images_r[i] for i in sample_indexes]
    sample_labels = [labels_input[i] for i in sample_indexes]

    # Run the "correct_pred" operation
    predicted = sess.run([correct_pred], feed_dict={x: sample_images})[0]

    # Print the real and predicted labels
    print(sample_labels)
    print(predicted)

    #predicted = sess.run([correct_pred], feed_dict={x: images_test})[0]
    # Calculate correct matches
    match_count = sum([int(y == y_) for y, y_ in zip(sample_labels, predicted)])
    a = np.float32(len(sample_labels))
    # Calculate the accuracy
    accuracy =np.float32(match_count)/np.float32(len(sample_labels))

    # Print the accuracy
    print"Accuracy:",accuracy

# Display the predictions and the ground truth visually.
fig = plt.figure(figsize=(10, 10))
for i in range(len(sample_images)):
    truth = sample_labels[i]
    prediction = predicted[i]
    plt.subplot(5, 2,1+i)
    plt.axis('off')
    plt.text(40, 10, "Truth:        {0}\nPrediction: {1}".format(truth, prediction),
             fontsize=12)
    plt.imshow(sample_images[i])

plt.show()

