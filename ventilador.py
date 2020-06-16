# -*- coding: utf-8 -*-
"""
Created on Sat Jun  6 23:28:10 2020
Ventilador
@author: caiom
"""
import numpy as np
import pandas as pd
import glob
from scipy.fftpack import fft
import math
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score
from skimage import util
from keras.utils import np_utils

# %% Defining some constants to be used throughout
# number os data points per set
data_points = 6660*3*100
# image width
img_w = 28
# image length
img_h = 28
# matrix used to hold final images
A = np.zeros((0, img_w, img_h))
B = np.zeros((0, img_w, img_h))
# image length when unidimensional
img_length = img_w*img_h
# number of data points used to build image
N = img_length*2
# images in each class
samples_per_class = (data_points)//N
# bitmap style
styleoff = "gray_r"
style1 = "viridis_r"
style2 = "inferno_r"
style3 = "cividis_r"

# %%
# reading csv and making dataframes
all_files = glob.glob("./ventilas_3s/off_3s/*.csv")

li = []

for filename in all_files:
    df = pd.read_csv(filename, index_col=None, header=0)
    li.append(df)

off = pd.concat(li, axis=0, ignore_index=True)

all_files = glob.glob("./ventilas_3s/speed1_3s/*.csv")

li = []

for filename in all_files:
    df = pd.read_csv(filename, index_col=None, header=0)
    li.append(df)

speed1 = pd.concat(li, axis=0, ignore_index=True)

all_files = glob.glob("./ventilas_3s/speed2_3s/*.csv")

li = []

for filename in all_files:
    df = pd.read_csv(filename, index_col=None, header=0)
    li.append(df)

speed2 = pd.concat(li, axis=0, ignore_index=True)

all_files = glob.glob("./ventilas_3s/speed3_3s/*.csv")

li = []

for filename in all_files:
    df = pd.read_csv(filename, index_col=None, header=0)
    li.append(df)

speed3 = pd.concat(li, axis=0, ignore_index=True)

# %% some statistics
avg_off = off.sum()/data_points
avg_speed1 = speed1.sum()/data_points
avg_speed2 = speed2.sum()/data_points
avg_speed3 = speed3.sum()/data_points
rms_off = np.sqrt(np.mean(off**2))
rms_speed1 = np.sqrt(np.mean(speed1**2))
rms_speed2 = np.sqrt(np.mean(speed2**2))
rms_speed3 = np.sqrt(np.mean(speed3**2))
# %%
speed1 = speed1.to_numpy()
speed2 = speed2.to_numpy()
speed3 = speed3.to_numpy()
speed1 = speed1.reshape(66600,)
speed2 = speed2.reshape(66600,)
speed3 = speed3.reshape(66600,)

# %%
M = N

slices = util.view_as_windows(speed1, window_shape=(M, ), step=M)
print(f'Signal shape: {speed1.shape}, Sliced signal shape: {slices.shape}')

for slice in slices:
    fftsl = np.abs(fft(slice)[:M // 2])
    T = []
    for x in range(0, len(fftsl)):
        T.append(abs(np.log2(fftsl[x]/img_length)))
    T = np.asarray(T)

    n = np.max(T)
    H = T/n
    H = (255*H).astype(np.uint8)

    H.shape = (1, H.size//img_h, img_h)
    A = np.insert(A, 0, H, axis=0)
    # plot each image
    # It = H
    # It.shape = (It.size//img_h, img_h)
    # plt.imshow(It, cmap=style1)
    # plt.show()

Ht = H
Ht.shape = (Ht.size//img_h, img_h)
plt.imshow(Ht, cmap=style1)
plt.show()

# %%
M = N

slices = util.view_as_windows(speed2, window_shape=(M, ), step=M)
print(f'Signal shape: {speed2.shape}, Sliced signal shape: {slices.shape}')

for slice in slices:
    fftsl = np.abs(fft(slice)[:M // 2])
    T = []
    for x in range(0, len(fftsl)):
        T.append(abs(np.log2(fftsl[x]/img_length)))
    T = np.asarray(T)

    n = np.max(T)
    H = T/n
    H = (255*H).astype(np.uint8)

    H.shape = (1, H.size//img_h, img_h)
    A = np.insert(A, 0, H, axis=0)
    # plot each image
    # It = H
    # It.shape = (It.size//img_h, img_h)
    # plt.imshow(It, cmap=style2)
    # plt.show()

Ht = H
Ht.shape = (Ht.size//img_h, img_h)
plt.imshow(Ht, cmap=style2)
plt.show()

# %%
M = N

slices = util.view_as_windows(speed3, window_shape=(M, ), step=M)
print(f'Signal shape: {speed3.shape}, Sliced signal shape: {slices.shape}')

for slice in slices:
    fftsl = np.abs(fft(slice)[:M // 2])
    T = []
    for x in range(0, len(fftsl)):
        T.append(abs(np.log2(fftsl[x]/img_length)))
    T = np.asarray(T)

    n = np.max(T)
    H = T/n
    H = (255*H).astype(np.uint8)

    H.shape = (1, H.size//img_h, img_h)
    A = np.insert(A, 0, H, axis=0)
    # plot each image
    It = H
    It.shape = (It.size//img_h, img_h)
    plt.imshow(It, cmap=style3)
    plt.show()

# Ht = H
# Ht.shape = (Ht.size//img_h, img_h)
# plt.imshow(Ht, cmap=style3)
# plt.show()

# %%
for i in range(0, samples_per_class):
    # fft
    # set y size
    y = off[(i*N):((i+1)*N)]
    # reshape
    y = y.reshape(N,)
    # get fft
    yf = fft(y)
    # abs and positive values
    ffty = np.abs(yf[0:N//2])

    # B2LS and append to S
    S = []
    for x in range(0, len(ffty)):
        S.append(abs(math.log2(ffty[x]/img_length)))

    # make S a numpy array
    S = np.asarray(S)

    # normalize into Img
    m = max(S)
    Img = S/m
    Img = (255*Img).astype(np.uint8)

    # make vector into 2d image and append to A
    Img.shape = (1, Img.size//img_h, img_h)
    B = np.insert(B, 0, Img, axis=0)

    # plot each image
    # It = Img
    # It.shape = (It.size//img_h, img_h)
    # plt.imshow(It, cmap=style1)
    # plt.show()

# plot last image
It = Img
It.shape = (It.size//img_h, img_h)
plt.imshow(It, cmap=style1)
plt.show()
# %%
for i in range(0, samples_per_class):
    # fft
    # set y size
    y = speed1[(i*N):((i+1)*N)]
    # reshape
    y = y.reshape(N,)
    # get fft
    yf = fft(y)
    # abs and positive values
    ffty = np.abs(yf[0:N//2])

    # B2LS and append to S
    S = []
    for x in range(0, len(ffty)):
        S.append(abs(math.log2(ffty[x]/img_length)))

    # make S a numpy array
    S = np.asarray(S)

    # normalize into Img
    m = max(S)
    Img = S/m
    Img = (255*Img).astype(np.uint8)

    # make vector into 2d image and append to A
    Img.shape = (1, Img.size//img_h, img_h)
    B = np.insert(B, 0, Img, axis=0)

    # plot each image
    # It = Img
    # It.shape = (It.size//img_h, img_h)
    # plt.imshow(It, cmap=style1)
    # plt.show()

# plot last image
It = Img
It.shape = (It.size//img_h, img_h)
plt.imshow(It, cmap=style1)
plt.show()

# %%
for i in range(0, samples_per_class):
    # fft
    # set y size
    y = speed2[(i*N):((i+1)*N)]
    # reshape
    y = y.reshape(N,)
    # get fft
    yf = fft(y)
    # abs and positive values
    ffty = np.abs(yf[0:N//2])

    # B2LS and append to S
    S = []
    for x in range(0, len(ffty)):
        S.append(abs(math.log2(ffty[x]/img_length)))

    # make S a numpy array
    S = np.asarray(S)

    # normalize into Img
    m = max(S)
    Img = S/m
    Img = (255*Img).astype(np.uint8)

    # make vector into 2d image and append to A
    Img.shape = (1, Img.size//img_h, img_h)
    B = np.insert(B, 0, Img, axis=0)

    # plot each image
    # It = Img
    # It.shape = (It.size//img_h, img_h)
    # plt.imshow(It, cmap=style2)
    # plt.show()

# plot last image
It = Img
It.shape = (It.size//img_h, img_h)
plt.imshow(It, cmap=style2)
plt.show()

# %%
for i in range(0, samples_per_class):
    # fft
    # set y size
    y = speed3[(i*N):((i+1)*N)]
    # reshape
    y = y.reshape(N,)
    # get fft
    yf = fft(y)
    # abs and positive values
    ffty = np.abs(yf[0:N//2])

    # B2LS and append to S
    S = []
    for x in range(0, len(ffty)):
        S.append(abs(math.log2(ffty[x]/img_length)))

    # make S a numpy array
    S = np.asarray(S)

    # normalize into Img
    m = max(S)
    Img = S/m
    Img = (255*Img).astype(np.uint8)

    # make vector into 2d image and append to A
    Img.shape = (1, Img.size//img_h, img_h)
    B = np.insert(B, 0, Img, axis=0)

    # plot each image
    # It = Img
    # It.shape = (It.size//img_h, img_h)
    # plt.imshow(It, cmap=style2)
    # plt.show()

# plot last image
It = Img
It.shape = (It.size//img_h, img_h)
plt.imshow(It, cmap=style3)
plt.show()
# %% Reshape A
A = A.reshape(A.shape[0], img_w, img_h, 1)
B = B.reshape(B.shape[0], img_w, img_h, 1)

# %% Appy labels to samples
# Label1 identifies only normal baseline and fault, two classes
# label1 = np.zeros(samples_per_class*3)
# label1[0:(samples_per_class)] = 1

# label2 identifies normal baseline and each specific fault at 2hp, six classes
label2 = np.zeros(samples_per_class*4)
label2[0:samples_per_class] = 1
label2[samples_per_class:samples_per_class*2] = 2
label2[samples_per_class*2:samples_per_class*3] = 3

label2 = np_utils.to_categorical(label2, 4)
# %% Build first CNN
# import modules
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from keras import models
from keras.optimizers import Adam

# define as sequential
model1 = models.Sequential()
# add first convolutional layer
model1.add(Conv2D(16, (3, 3), activation='relu',
                  input_shape=(img_w, img_h, 1)))
# add first max pooling layer
model1.add(MaxPooling2D(pool_size=(3, 3), strides=(2, 2)))
# add second convolutional layer
model1.add(Conv2D(32, (3, 3), activation='relu'))
# add second max pooling layer
model1.add(MaxPooling2D(pool_size=(3, 3), strides=(2, 2)))
# flatten before mlp
model1.add(Flatten())
# add fully connected wih 128 neurons and relu activation
model1.add(Dense(128, activation='relu'))
# output six classes with softmax activtion
model1.add(Dense(4, activation='softmax'))

# print CNN info
model1.summary()
# compile CNN and define its functions
model1.compile(loss='categorical_crossentropy', optimizer=Adam(),
               metrics=['accuracy'])

# %% Same as above for another, simpler CNN model
# define as sequential
model2 = models.Sequential()
# add first convolutional layer
model2.add(Conv2D(2, (2, 2), activation='relu',
                  input_shape=(img_w, img_h, 1)))
# add first max pooling layer
model2.add(MaxPooling2D(pool_size=(3, 3), strides=(2, 2)))
# flatten befor MLP
model2.add(Flatten())
# add fully connected wih 8 neurons and relu activation
model2.add(Dense(8, activation='relu'))
# output six classes with softmax activtion
model2.add(Dense(3, activation='softmax'))

# print CNN info
model2.summary()
# compile CNN and define its functions
model2.compile(loss='categorical_crossentropy', optimizer=Adam(),
               metrics=['accuracy'])

# %% Separate classes, labels, train and test
# X_train, X_test, y_train, y_test =train_test_split(A, label1, test_size=0.25)
X_train, X_test, y_train, y_test = train_test_split(A, label2, test_size=0.25)
X_train = X_train.astype('uint8')
X_test = X_test.astype('uint8')

# %% Train CNN model1

model1.fit(X_train, y_train, batch_size=1, nb_epoch=20,
           validation_data=(X_test, y_test))
# %% Results
# Make inference
# Predict and normalize predictions into 0s and 1s
predictions = model1.predict(X_test)
predictions1 = (predictions > 0.5)
# Find accuracy of inference
accuracy = accuracy_score(y_test, predictions1)
print(accuracy)
# calculate confusion matrix and print it
matrix = confusion_matrix(y_test.argmax(axis=1), predictions1.argmax(axis=1))
print(matrix)
# %%
# Use evaluate to test, just another way to do the same thing
result = model1.evaluate(X_test, y_test)
print(result)

# %% Train CNN model2
model2.fit(X_train, y_train, batch_size=1, nb_epoch=20,
           validation_data=(X_test, y_test))

# %% Results
# Make inference
# Predict and normalize predictions into 0s and 1s
predictions = model2.predict(X_test)
predictions = (predictions > 0.5)
# Find accuracy of inference
accuracy = accuracy_score(y_test, predictions)
print(accuracy)
# calculate confusion matrix and print it
matrix = confusion_matrix(y_test.argmax(axis=1), predictions.argmax(axis=1))
print(matrix)

# Use evaluate to test, just another way to do the same thing
result = model2.evaluate(X_test, y_test)
print(result)
confusion_matrix(y_test.argmax(axis=1), predictions.argmax(axis=1))

# %% Separate classes, labels, train and test
# X_train, X_test, y_train, y_test =train_test_split(A, label1, test_size=0.25)
X_train, X_test, y_train, y_test = train_test_split(B, label2, test_size=0.25)
X_train = X_train.astype('uint8')
X_test = X_test.astype('uint8')

# %% Train CNN model1
model1.fit(X_train, y_train, batch_size=1, nb_epoch=20,
           validation_data=(X_test, y_test))
# %% Results
# Make inference
# Predict and normalize predictions into 0s and 1s
predictions = model1.predict(X_test)
predictions1 = (predictions > 0.5)
# Find accuracy of inference
accuracy = accuracy_score(y_test, predictions1)
print(accuracy)
# calculate confusion matrix and print it
matrix = confusion_matrix(y_test.argmax(axis=1), predictions1.argmax(axis=1))
print(matrix)
# %%
# Use evaluate to test, just another way to do the same thing
result = model1.evaluate(X_test, y_test)
print(result)

# %% Train CNN model2
model2.fit(X_train, y_train, batch_size=1, nb_epoch=20,
           validation_data=(X_test, y_test))

# %% Results
# Make inference
# Predict and normalize predictions into 0s and 1s
predictions = model2.predict(X_test)
predictions = (predictions > 0.5)
# Find accuracy of inference
accuracy = accuracy_score(y_test, predictions)
print(accuracy)
# calculate confusion matrix and print it
matrix = confusion_matrix(y_test.argmax(axis=1), predictions.argmax(axis=1))
print(matrix)

# Use evaluate to test, just another way to do the same thing
result = model2.evaluate(X_test, y_test)
print(result)
confusion_matrix(y_test.argmax(axis=1), predictions.argmax(axis=1))

