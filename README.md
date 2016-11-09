
# Traffic Sign Classification with Keras

Keras exists to make coding deep neural networks simpler. To demonstrate just how easy it is, you’re going to use Keras to build a convolutional neural network in a few dozen lines of code.

You’ll be connecting the concepts from the previous lessons to the methods that Keras provides.

## Dataset

The network you'll build with Keras is similar to the example that you can find in Keras’s GitHub repository that builds out a [convolutional neural network for MNIST](https://github.com/fchollet/keras/blob/master/examples/mnist_cnn.py). 

However, instead of using the [MNIST](http://yann.lecun.com/exdb/mnist/) dataset, you're going to use the [German Traffic Sign Recognition Benchmark](http://benchmark.ini.rub.de/?section=gtsrb&subsection=news) dataset that you've used previously.

You can download pickle files with sanitized traffic sign data here.

## Overview

Here are the steps you'll take to build the network:

1. First load the data.
2. Build a feedforward neural network to classify traffic signs.
3. Build a convolutional neural network to classify traffic signs.

Keep an eye on the network’s accuracy over time. Once the accuracy reaches the 98% range, you can be confident that you’ve built and trained an effective model.

## Load the Data

Start by importing the data from the pickle file.


```python
# TODO: Implement load the data here.
import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from six.moves import cPickle as pickle
from six.moves import range

# Load train data
with open("train.p", 'rb') as F:
    save = pickle.load(F)
    X_train = save['features']
    y_train = save['labels']

# Load train data
with open("test.p", 'rb') as F:
    save = pickle.load(F)
    X_test = save['features']
    y_test = save['labels']    
    
# STOP: Do not change the tests below. Your implementation should pass these tests. 
assert(X_train.shape[0] == y_train.shape[0]), "The number of images is not equal to the number of labels."
assert(X_train.shape[1:] == (32,32,3)), "The dimensions of the images are not 32 x 32 x 3."
assert(X_test.shape[0] == y_test.shape[0]), "The number of images is not equal to the number of labels."
assert(X_test.shape[1:] == (32,32,3)), "The dimensions of the images are not 32 x 32 x 3."

print("Ok")
```

    Ok


## Normalize the data

Now that you've loaded the training data, normalize the input so that it has a mean of 0 and a range between -0.5 and 0.5.


```python
# TODO: Implement data normalization here.

# Function for normalization
def normalize(image_data):
    a = -0.5
    b = 0.5
    image_min = 0
    image_max = 255    
    return a + (((image_data - image_min)*(b - a))/(image_max - image_min))

# normalizing train data
X_train = normalize(X_train)
X_test = normalize(X_test)

# Image shape
input_shape = X_train[0].shape

# STOP: Do not change the tests below. Your implementation should pass these tests. 
assert(round(np.mean(X_train)) == 0), "The mean of the input data is: %f" % np.mean(X_train)
assert(np.min(X_train) == -0.5 and np.max(X_train) == 0.5), "The range of the input data is: %.1f to %.1f" % (np.min(X_train), np.max(X_train))
```

## Build a Two-Layer Feedfoward Network

The code you've written so far is for data processing, not specific to Keras. Here you're going to build Keras-specific code.

Build a two-layer feedforward neural network, with 128 neurons in the fully-connected hidden layer. 

To get started, review the Keras documentation about [models](https://keras.io/models/sequential/) and [layers](https://keras.io/layers/core/).

The Keras example of a [Multi-Layer Perceptron](https://github.com/fchollet/keras/blob/master/examples/mnist_mlp.py) network is similar to what you need to do here. Use that as a guide, but keep in mind that there are a number of differences.


```python
# TODO: Build a two-layer feedforward neural network with Keras here.
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D
from keras.optimizers import SGD, Adam, RMSprop
from keras.utils import np_utils
from keras import backend as K
np.random.seed(1337)

features_count = input_shape[0] * input_shape[1] * input_shape[2]

batch_size = 128
nb_classes = 43
nb_epoch = 20

print(X_train.shape)
X_train = X_train.reshape(X_train.shape[0], features_count)
X_test = X_test.reshape(X_test.shape[0], features_count)
X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
print(X_train.shape[0], 'train samples')
print(X_test.shape[0], 'test samples')

# convert class vectors to binary class matrices
Y_train = np_utils.to_categorical(y_train, nb_classes)
Y_test = np_utils.to_categorical(y_test, nb_classes)

model = Sequential()

model.add(Dense(512, input_shape=(features_count,), name="hidden1"))
model.add(Activation('relu'))
model.add(Dropout(0.2))
model.add(Dense(512))
model.add(Activation('relu'))
model.add(Dropout(0.2))
model.add(Dense(nb_classes))
model.add(Activation('softmax', name="output"))

model.summary()

# STOP: Do not change the tests below. Your implementation should pass these tests.
assert(model.get_layer(name="hidden1").input_shape == (None, 32*32*3)), "The input shape is: %s" % model.get_layer(name="hidden1").input_shape
assert(model.get_layer(name="output").output_shape == (None, 43)), "The output shape is: %s" % model.get_layer(name="output").output_shape 

print("ok")
```

    (39209, 32, 32, 3)
    39209 train samples
    12630 test samples
    ____________________________________________________________________________________________________
    Layer (type)                     Output Shape          Param #     Connected to                     
    ====================================================================================================
    hidden1 (Dense)                  (None, 512)           1573376     dense_input_17[0][0]             
    ____________________________________________________________________________________________________
    activation_55 (Activation)       (None, 512)           0           hidden1[0][0]                    
    ____________________________________________________________________________________________________
    dropout_39 (Dropout)             (None, 512)           0           activation_55[0][0]              
    ____________________________________________________________________________________________________
    dense_40 (Dense)                 (None, 512)           262656      dropout_39[0][0]                 
    ____________________________________________________________________________________________________
    activation_56 (Activation)       (None, 512)           0           dense_40[0][0]                   
    ____________________________________________________________________________________________________
    dropout_40 (Dropout)             (None, 512)           0           activation_56[0][0]              
    ____________________________________________________________________________________________________
    dense_41 (Dense)                 (None, 43)            22059       dropout_40[0][0]                 
    ____________________________________________________________________________________________________
    output (Activation)              (None, 43)            0           dense_41[0][0]                   
    ====================================================================================================
    Total params: 1858091
    ____________________________________________________________________________________________________
    ok


## Train the Network
Compile and train the network for 2 epochs. [Use the `adam` optimizer, with `categorical_crossentropy` loss.](https://keras.io/models/sequential/)

Hint 1: In order to use categorical cross entropy, you will need to [one-hot encode the labels](https://github.com/fchollet/keras/blob/master/keras/utils/np_utils.py).

Hint 2: In order to pass the input images to the fully-connected hidden layer, you will need to [reshape the input](https://github.com/fchollet/keras/blob/master/examples/mnist_mlp.py).

Hint 3: Keras's `.fit()` method returns a `History.history` object, which the tests below use. Save that to a variable named `history`.


```python
# TODO: Compile and train the model here.
model.compile(loss='categorical_crossentropy',
              optimizer=Adam(),
              metrics=['accuracy'])

history = model.fit(X_train, Y_train,
                    batch_size=batch_size, nb_epoch=nb_epoch,
                    verbose=1, validation_data=(X_test, Y_test))
score = model.evaluate(X_test, Y_test, verbose=0)
print('Test score:', score[0])
print('Test accuracy:', score[1])

# STOP: Do not change the tests below. Your implementation should pass these tests.
assert(history.history['acc'][0] > 0.5), "The training accuracy was: %.3f" % history.history['acc']
```

    Train on 39209 samples, validate on 12630 samples
    Epoch 1/20
    39209/39209 [==============================] - 26s - loss: 1.6854 - acc: 0.5244 - val_loss: 1.1533 - val_acc: 0.6635
    Epoch 2/20
    39209/39209 [==============================] - 26s - loss: 0.8403 - acc: 0.7489 - val_loss: 0.9606 - val_acc: 0.7344
    Epoch 3/20
    39209/39209 [==============================] - 25s - loss: 0.6303 - acc: 0.8087 - val_loss: 0.9476 - val_acc: 0.7490
    Epoch 4/20
    39209/39209 [==============================] - 26s - loss: 0.5278 - acc: 0.8402 - val_loss: 0.9959 - val_acc: 0.7492
    Epoch 5/20
    39209/39209 [==============================] - 24s - loss: 0.5029 - acc: 0.8472 - val_loss: 0.9086 - val_acc: 0.7787
    Epoch 6/20
    39209/39209 [==============================] - 28s - loss: 0.4322 - acc: 0.8676 - val_loss: 0.8956 - val_acc: 0.7989
    Epoch 7/20
    39209/39209 [==============================] - 26s - loss: 0.4253 - acc: 0.8709 - val_loss: 0.9133 - val_acc: 0.7862
    Epoch 8/20
    39209/39209 [==============================] - 27s - loss: 0.4154 - acc: 0.8716 - val_loss: 0.8495 - val_acc: 0.8109
    Epoch 9/20
    39209/39209 [==============================] - 24s - loss: 0.3623 - acc: 0.8884 - val_loss: 1.0281 - val_acc: 0.7826
    Epoch 10/20
    39209/39209 [==============================] - 26s - loss: 0.3562 - acc: 0.8892 - val_loss: 0.9224 - val_acc: 0.8048
    Epoch 11/20
    39209/39209 [==============================] - 26s - loss: 0.3618 - acc: 0.8891 - val_loss: 0.9022 - val_acc: 0.7999
    Epoch 12/20
    39209/39209 [==============================] - 25s - loss: 0.3329 - acc: 0.8978 - val_loss: 0.9657 - val_acc: 0.7998
    Epoch 13/20
    39209/39209 [==============================] - 24s - loss: 0.3377 - acc: 0.8966 - val_loss: 1.0911 - val_acc: 0.7876
    Epoch 14/20
    39209/39209 [==============================] - 26s - loss: 0.3322 - acc: 0.8974 - val_loss: 0.8890 - val_acc: 0.8184
    Epoch 15/20
    39209/39209 [==============================] - 26s - loss: 0.3282 - acc: 0.9003 - val_loss: 0.9190 - val_acc: 0.8243
    Epoch 16/20
    39209/39209 [==============================] - 28s - loss: 0.3079 - acc: 0.9055 - val_loss: 0.9846 - val_acc: 0.8059
    Epoch 17/20
    39209/39209 [==============================] - 26s - loss: 0.2997 - acc: 0.9090 - val_loss: 1.1228 - val_acc: 0.7850
    Epoch 18/20
    39209/39209 [==============================] - 25s - loss: 0.2839 - acc: 0.9132 - val_loss: 1.1842 - val_acc: 0.7864
    Epoch 19/20
    39209/39209 [==============================] - 26s - loss: 0.3055 - acc: 0.9070 - val_loss: 1.0941 - val_acc: 0.7919
    Epoch 20/20
    39209/39209 [==============================] - 23s - loss: 0.2805 - acc: 0.9128 - val_loss: 1.0050 - val_acc: 0.8124
    Test score: 1.00495262031
    Test accuracy: 0.812351543924


## Validate the Network
Split the training data into a training and validation set.

Measure the [validation accuracy](https://keras.io/models/sequential/) of the network after two training epochs.

Hint: [Use the `train_test_split()` method](http://scikit-learn.org/stable/modules/generated/sklearn.model_selection.train_test_split.html) from scikit-learn.


```python
# TODO: Split some of the training data into a validation dataset.
# TODO: Compile and train the model to measure validation accuracy.
from sklearn.model_selection import train_test_split
from sklearn.utils import resample

# Get randomized datasets for training and validation
X_train, X_val, y_train, y_val = train_test_split(
    X_train,
    y_train,
    test_size=0.25,
    random_state=832289)

# STOP: Do not change the tests below. Your implementation should pass these tests.
assert(round(X_train.shape[0] / float(X_val.shape[0])) == 3), "The training set is %.3f times larger than the validation set." % X_train.shape[0] / float(X_val.shape[0])
assert(history.history['val_acc'][0] > 0.6), "The validation accuracy is: %.3f" % history.history['val_acc'][0]
```

**Validation Accuracy**: (0.812351543924)

## Congratulations
You've built a feedforward neural network in Keras!

Don't stop here! Next, you'll add a convolutional layer to drive.py.

## Convolutions
Build a new network, similar to your existing network. Before the hidden layer, add a 3x3 [convolutional layer](https://keras.io/layers/convolutional/#convolution2d) with 32 filters and valid padding.

Then compile and train the network.

Hint 1: The Keras example of a [convolutional neural network](https://github.com/fchollet/keras/blob/master/examples/mnist_cnn.py) for MNIST would be a good example to review.

Hint 2: Now that the first layer of the network is a convolutional layer, you no longer need to reshape the input images before passing them to the network. You might need to reload your training data to recover the original shape.

Hint 3: Add a [`Flatten()` layer](https://keras.io/layers/core/#flatten) between the convolutional layer and the fully-connected hidden layer.


```python
# TODO: Re-construct the network and add a convolutional layer before the first fully-connected layer.
# TODO: Compile and train the model.

np.random.seed(1337)

features_count = input_shape[0] * input_shape[1] * input_shape[2]

batch_size = 128
nb_classes = 43
nb_epoch = 12

# number of convolutional filters to use
nb_filters = 32
# size of pooling area for max pooling
pool_size = (2, 2)
# convolution kernel size
kernel_size = (3, 3)

X_train = X_train.reshape(X_train.shape[0], input_shape[0], input_shape[1], input_shape[2])
X_val = X_val.reshape(X_val.shape[0], input_shape[0], input_shape[1], input_shape[2])
X_test = X_test.reshape(X_test.shape[0], input_shape[0], input_shape[1], input_shape[2])

# convert class vectors to binary class matrices
Y_train = np_utils.to_categorical(y_train, nb_classes)
Y_val = np_utils.to_categorical(y_val, nb_classes)
Y_test = np_utils.to_categorical(y_test, nb_classes)

# Constructing model with convolutional layers
model = Sequential()

model.add(Convolution2D(nb_filters, kernel_size[0], kernel_size[1],
                        border_mode='valid',
                        input_shape=input_shape))
model.add(Activation('relu'))
model.add(Convolution2D(nb_filters, kernel_size[0], kernel_size[1]))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=pool_size))
model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(128))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(nb_classes))
model.add(Activation('softmax'))

model.compile(loss='categorical_crossentropy',
              optimizer='adadelta',
              metrics=['accuracy'])

model.fit(X_train, Y_train, batch_size=batch_size, nb_epoch=nb_epoch,
          verbose=1, validation_data=(X_val, Y_val))
score = model.evaluate(X_test, Y_test, verbose=0)
```

    Train on 29406 samples, validate on 9803 samples
    Epoch 1/12
    29406/29406 [==============================] - 118s - loss: 2.3964 - acc: 0.3489 - val_loss: 0.9896 - val_acc: 0.7192
    Epoch 2/12
    29406/29406 [==============================] - 118s - loss: 0.8351 - acc: 0.7481 - val_loss: 0.3320 - val_acc: 0.9269
    Epoch 3/12
    29406/29406 [==============================] - 117s - loss: 0.4837 - acc: 0.8579 - val_loss: 0.1959 - val_acc: 0.9567
    Epoch 4/12
    29406/29406 [==============================] - 104s - loss: 0.3630 - acc: 0.8934 - val_loss: 0.1536 - val_acc: 0.9661
    Epoch 5/12
    29406/29406 [==============================] - 132s - loss: 0.2931 - acc: 0.9145 - val_loss: 0.1283 - val_acc: 0.9700
    Epoch 6/12
    29406/29406 [==============================] - 215s - loss: 0.2481 - acc: 0.9280 - val_loss: 0.1080 - val_acc: 0.9793
    Epoch 7/12
    29406/29406 [==============================] - 113s - loss: 0.2125 - acc: 0.9386 - val_loss: 0.0966 - val_acc: 0.9781
    Epoch 8/12
    29406/29406 [==============================] - 106s - loss: 0.1859 - acc: 0.9455 - val_loss: 0.0812 - val_acc: 0.9813
    Epoch 9/12
    29406/29406 [==============================] - 106s - loss: 0.1659 - acc: 0.9516 - val_loss: 0.0752 - val_acc: 0.9835
    Epoch 10/12
    29406/29406 [==============================] - 106s - loss: 0.1508 - acc: 0.9556 - val_loss: 0.0813 - val_acc: 0.9783
    Epoch 11/12
    29406/29406 [==============================] - 106s - loss: 0.1369 - acc: 0.9604 - val_loss: 0.0600 - val_acc: 0.9857
    Epoch 12/12
    29406/29406 [==============================] - 106s - loss: 0.1256 - acc: 0.9633 - val_loss: 0.0570 - val_acc: 0.9862
    Test score: 0.251709717874
    Test accuracy: 0.939825811579



    ---------------------------------------------------------------------------

    AssertionError                            Traceback (most recent call last)

    <ipython-input-74-e1339cf096f5> in <module>()
         56 
         57 # STOP: Do not change the tests below. Your implementation should pass these tests.
    ---> 58 assert(history.history['val_acc'][0] > 0.9), "The validation accuracy is: %.3f" % history.history['val_acc'][0]
    

    AssertionError: The validation accuracy is: 0.663



```python
print('Test score:', score[0])
print('Test accuracy:', score[1])

# STOP: Do not change the tests below. Your implementation should pass these tests.
assert(history.history['val_acc'][0] > 0.9), "The validation accuracy is: %.3f" % history.history['val_acc'][0]
```

    Test score: 0.251709717874
    Test accuracy: 0.939825811579
    [0.6634996042021285, 0.73436262868079327, 0.74901029297216304, 0.74916864606188294, 0.7787015043735882, 0.79889152802273289, 0.78622327804187886, 0.8109263659278646, 0.7825811560344621, 0.80482977018975499, 0.79992082360615735, 0.79976247018608704, 0.78756927933281407, 0.81836896262655912, 0.82430720519944023, 0.80593824245017953, 0.7850356296235762, 0.78638163083900281, 0.79192399033835648, 0.81235154403737875]



    ---------------------------------------------------------------------------

    AssertionError                            Traceback (most recent call last)

    <ipython-input-77-433ffb8fe848> in <module>()
          3 print(history.history['val_acc'])
          4 # STOP: Do not change the tests below. Your implementation should pass these tests.
    ----> 5 assert(history.history['val_acc'][0] > 0.9), "The validation accuracy is: %.3f" % history.history['val_acc'][0]
    

    AssertionError: The validation accuracy is: 0.663


**Validation Accuracy**: (fill in here)

## Pooling
Re-construct your network and add a 2x2 [pooling layer](https://keras.io/layers/pooling/#maxpooling2d) immediately following your convolutional layer.

Then compile and train the network.


```python
# TODO: Re-construct the network and add a pooling layer after the convolutional layer.
# TODO: Compile and train the model.

# STOP: Do not change the tests below. Your implementation should pass these tests.
assert(history.history['val_acc'][0] > 0.9), "The validation accuracy is: %.3f" % history.history['val_acc'][0]
```

**Validation Accuracy**: (fill in here)

## Dropout
Re-construct your network and add [dropout](https://keras.io/layers/core/#dropout) after the pooling layer. Set the dropout rate to 50%.


```python
# TODO: Re-construct the network and add dropout after the pooling layer.
# TODO: Compile and train the model.

# STOP: Do not change the tests below. Your implementation should pass these tests.
assert(history.history['val_acc'][0] > 0.9), "The validation accuracy is: %.3f" % history.history['val_acc'][0]
```


```python
**Validation Accuracy**: (fill in here)
```

## Optimization
Congratulations! You've built a neural network with convolutions, pooling, dropout, and fully-connected layers, all in just a few lines of code.

Have fun with the model and see how well you can do! Add more layers, or regularization, or different padding, or batches, or more training epochs.

What is the best validation accuracy you can achieve?


```python

```

**Best Validation Accuracy:** (fill in here)

## Testing
Once you've picked out your best model, it's time to test it.

Load up the test data and use the [`evaluate()` method](https://keras.io/models/model/#evaluate) to see how well it does.

Hint 1: After you load your test data, don't forget to normalize the input and one-hot encode the output, so it matches the training data.

Hint 2: The `evaluate()` method should return an array of numbers. Use the `metrics_names()` method to get the labels.


```python
with open('./test.p', mode='rb') as f:
    test = pickle.load(f)
    
X_test = test['features']
y_test = test['labels']
X_test = X_test.astype('float32')
X_test /= 255
X_test -= 0.5
Y_test = np_utils.to_categorical(y_test, 43)

model.evaluate(X_test, Y_test)
```

    12630/12630 [==============================] - 16s    





    [0.2517097149292038, 0.93982581157865552]



**Test Accuracy:** (fill in here)

## Summary
Keras is a great tool to use if you want to quickly build a neural network and evaluate performance.
