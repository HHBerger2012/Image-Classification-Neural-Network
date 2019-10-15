#https://developer.ibm.com/articles/image-recognition-challenge-with-tensorflow-and-keras-pt1/
#turn to black and white first 

import tensorflow as tf
import skimage.io as io
import matplotlib.pyplot as plt
from skimage.color import rgb2gray
from skimage.transform import rescale
import sklearn.model_selection as sk
import numpy as np
import os
import time
import random

cardboard_path = '/Users/henryberger/Desktop/Summer_Project/dataset-resized/cardboard/'
glass_path = '/Users/henryberger/Desktop/Summer_Project/dataset-resized/glass/'
metal_path = '/Users/henryberger/Desktop/Summer_Project/dataset-resized/metal/'
paper_path = '/Users/henryberger/Desktop/Summer_Project/dataset-resized/paper/'
plastic_path = '/Users/henryberger/Desktop/Summer_Project/dataset-resized/plastic/'
trash_path = '/Users/henryberger/Desktop/Summer_Project/dataset-resized/trash/'


print(tf.VERSION)

def load_paths(path1, path2, path3, path4, path5, path6):
    paths = []
    for i in (os.listdir(path1)):
        input_path = os.path.join(path1, i)
        paths.append(input_path)
    for i in (os.listdir(path2)):
        input_path = os.path.join(path2, i)
        paths.append(input_path)
    for i in (os.listdir(path3)):
        input_path = os.path.join(path3, i)
        paths.append(input_path)
    for i in (os.listdir(path4)):
        input_path = os.path.join(path4, i)
        paths.append(input_path)
    for i in (os.listdir(path5)):
        input_path = os.path.join(path5, i)
        paths.append(input_path)
    for i in (os.listdir(path6)):
        input_path = os.path.join(path6, i)
        paths.append(input_path)
    
    print('Loaded Paths\n')
    return paths


#make array, not string
def load_images(paths):
    images = []
    labels = []
    for i in (paths):
        temp_image = io.imread(i)
        image = rescale(temp_image, 0.5, anti_aliasing=False)
        gray_image = rgb2gray(image)
        if(i[58:63] == 'cardb'):
            images.append(gray_image)
            labels.append([1,0,0,0,0,0])
        elif(i[58:63] == 'glass'):
            images.append(gray_image)
            labels.append([0,1,0,0,0,0])
        elif(i[58:63] == 'metal'):
            images.append(gray_image)
            labels.append([0,0,1,0,0,0])
        elif(i[58:63] == 'paper'):
            images.append(gray_image)
            labels.append([0,0,0,1,0,0])
        elif(i[58:63] == 'plast'):
            images.append(gray_image)
            labels.append([0,0,0,0,1,0])
        elif(i[58:63] == 'trash'):
            images.append(gray_image)
            labels.append([0,0,0,0,0,1])
        
    print('Loaded Images\n')
    return (np.asarray(images), np.asarray(labels))
        
        



all_paths = load_paths(cardboard_path, glass_path, metal_path, paper_path, plastic_path, trash_path)
random.shuffle(all_paths)
(all_images, all_labels) = load_images(all_paths) 
all_labels = np.expand_dims(all_labels,1)

X_train, X_test, y_train, y_test = sk.train_test_split(all_images,
                                                    all_labels,
                                                    test_size=0.30,
                                                    random_state=42)

#Helper
  
#Initial Weight

def init_weights(shape): 
    init_random_dist = tf.truncated_normal(shape, stddev = .1)
    return tf.Variable(init_random_dist)

#Initial Bias
    
def init_bias(shape):
    init_bias_vals = tf.constant(0.1, shape = shape)
    return tf.Variable(init_bias_vals)
    
#Conv2d

def conv2d(x, W):
     # x --> [batch,H,W,Channels]
     # W --> [filter H, filter W, Channels IN, Channe;s OUT]
     return tf.nn.conv2d(x,W, strides = [1,1,1,1], padding = 'SAME')

#Pooling 
    
def max_pool_2by2(x):
    # x --> {batch,H,W,Channels]
    return tf.nn.max_pool(x, ksize=[1,2,2,1],strides=[1,2,2,1],padding = 'SAME')

#Convolutional Layer
def convolutional_layer(input_x, shape):
    W = init_weights(shape)
    b = init_bias([shape[3]])
    return tf.nn.relu(conv2d(input_x,W)+b)

#Normal (Fully Connected)
    
def normal_full_layer(input_layer, size):
    input_size = int(input_layer.shape[1])
    W = init_weights([input_size, size])
    b = init_bias([size])
    return tf.matmul(input_layer, W) + b 
    
#Placeholders
    
x = tf.placeholder(tf.float32, shape = [None, 192,256], name = 'x')
y_true = tf.placeholder(tf.float32,shape=[None,1,6], name = 'y_true')

# layers

x_image = tf.reshape(x,[-1, 256, 192, 3])

convo_1 = convolutional_layer(x_image, shape = [5,5,1,36])
convo_1_pooling = max_pool_2by2(convo_1)

print(convo_1.shape)

convo_2 = convolutional_layer(convo_1_pooling, shape = [5,5,36,64])
convo_2_pooling = max_pool_2by2(convo_2)

print(convo_2.shape)

convo_2_flat = tf.reshape(convo_2_pooling, [-1, 192*256*64])
full_layer_one = tf.nn.relu(normal_full_layer(convo_2_flat, 1024))

print(convo_2_flat.shape)

# DROPOUT
hold_prob = tf.placeholder(tf.float32)
full_one_dropout = tf.nn.dropout(full_layer_one,keep_prob=hold_prob)
y_pred = normal_full_layer(full_one_dropout,6)

# LOSS FUNCTION
cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_true,logits=y_pred))

# OPTIMIZER
optimizer = tf.train.AdamOptimizer(learning_rate=0.001)
train = optimizer.minimize(cross_entropy)

init = tf.global_variables_initializer()

steps = 1000
with tf.Session() as sess:
    sess.run(init)

    for i in range(steps):
        total_batches = int(len(X_train) / 50)
        x_batches = np.array_split(X_train, total_batches)
        y_batches = np.array_split(y_train, total_batches)
        
        for j in range(total_batches):
            batch_x, batch_y = x_batches[j], y_batches[j]
            sess.run(train,feed_dict={x:batch_x, y_true:batch_y, hold_prob:0.5})
            
            if i%100 == 0:
                print("ON STEP: {}".format(i))
                print("ACCURACY: ")
                matches = tf.equal(tf.argmax(y_pred,1),tf.argmax(y_true,1))

                acc = tf.reduce_mean(tf.cast(matches,tf.float32))

                print(sess.run(acc,feed_dict={x:all_images,y_true:all_labels,hold_prob:1.0}))
                print("\n")


        

