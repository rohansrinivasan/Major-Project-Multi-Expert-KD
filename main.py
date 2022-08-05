# This file imports all the necessary libraries
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from scipy import signal
from keras.models import Sequential, load_model
from keras.layers import TimeDistributed,Conv1D,AveragePooling1D,Flatten,LSTM,Dense,BatchNormalization,Dropout
from keras.callbacks import EarlyStopping
from sklearn.metrics import accuracy_score,classification_report
from sklearn.metrics import confusion_matrix
import seaborn as sn
from keras.utils.vis_utils import plot_model
from tensorflow.keras.utils import to_categorical
from sklearn.datasets import make_blobs
from sklearn.metrics import accuracy_score
from keras.models import Sequential
from keras.layers import Dense
from matplotlib import pyplot
from numpy import mean
from numpy import std
from numpy import array
from numpy import argmax
from numpy import tensordot
from numpy.linalg import norm
from itertools import product
from pickle import load
from numpy import array
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils.vis_utils import plot_model
from keras.models import Model
from keras.layers import Input
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers import Dropout
from keras.layers import Embedding
from keras.layers.convolutional import Conv1D
from keras.layers.convolutional import MaxPooling1D
from keras.layers.merge import concatenate
 
# Loads the dataset used from drive
# Dataset consists of three modalities, i.e. Surface Electro-myogram or sEMG, tri-axis gyroscope and tri-axis accelerometer.
# Signals were captured using six Delsys wireless sensors, consisting of one sEMG sensor and one IMU containing a tri-axis accelerometer and a tri-axis gyroscope each. 

from google.colab import drive
drive.mount("/content/drive")
file_path = r'/content/drive/MyDrive/HGR_DL/'
emg = np.load(file_path+'datagen2_emgDom.npy');
acc = np.load(file_path+'datagen2_accDom.npy');
gyr = np.load(file_path+'datagen2_gyrDom.npy');
y_true = np.load(file_path+'datagen2_y_true.npy');
y=np.vstack((y_true,range(0,len(y_true)))).transpose();
chN=3;ax=3; 
seglenE = 3000; #int(np.round(1.25*fs[0])); #number of samples to downsample to 
seglenA = 400; #int(np.round(1.25*fs[1])); #number of samples to downsample to 
n_steps=10;n_lengthE=300; n_lengthA=40;

# sEMG Signal
emg = emg.reshape((emg.shape[0],chN, seglenE));
emg1 = signal.resample(emg, seglenA, t=None, axis=2);
emg = np.transpose(emg,axes=(0,2,1));
emg1 = np.transpose(emg1,axes=(0,2,1));

# Accelerometer Signal
acc = acc.reshape((acc.shape[0],chN*ax, seglenA));
acc = np.transpose(acc,axes=(0,2,1));

# Gyroscope Signal
gyr = gyr.reshape((gyr.shape[0],chN*ax, seglenA));
gyr = np.transpose(gyr,axes=(0,2,1));

# Making a single feature matrix of all three modalities with 5000 samples of each

X = np.concatenate((emg1,acc,gyr),axis=2)
# Reshape
X = X.reshape((X.shape[0],n_steps, n_lengthA,chN+chN*ax*2));
emg = emg.reshape((emg.shape[0],n_steps, n_lengthE,chN));
acc = acc.reshape((acc.shape[0],n_steps, n_lengthA,chN*ax));
gyr = gyr.reshape((gyr.shape[0],n_steps, n_lengthA,chN*ax));

print(X.shape)
print(emg.shape)
print(acc.shape)
print(gyr.shape)

# Reshaped data is split 70-30 where 70 is for the train split and 30 is for the test split.

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1, stratify=y_true)
emg_train, emg_test, y_train, y_test = train_test_split(emg, y, test_size=0.3, random_state=1, stratify=y_true)
acc_train, acc_test, y_train, y_test = train_test_split(acc, y, test_size=0.3, random_state=1, stratify=y_true)
gyr_train, gyr_test, y_train, y_test = train_test_split(gyr, y, test_size=0.3, random_state=1, stratify=y_true)
train_idx = y_train[:,1];   y_train = y_train[:,0]
test_idx = y_test[:,1];     y_test = y_test[:,0]
print(emg_train.shape)
print(acc_train.shape)
print(gyr_train.shape)
print(X_train.shape)
print(y_train.shape)

# Convert labels to categorical
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)
y_test1 = np.argmax(y_test, axis=1)
X_test1 = np.argmax(X_test, axis=1)

from keras.optimizers import adam_v2
n_outputs = y_train.shape[1]
verbose, epochs, batch_size = 1, 100, 64 #0, 15, 50
adam = adam_v2.Adam(lr=0.002)
es = EarlyStopping(monitor = 'val_accuracy',min_delta = 0.0002, patience = 5, verbose = 1,restore_best_weights = True)

#%  multiple input model
n_length, n_features= X_train.shape[2],X_train.shape[3]
n_emg_length, n_emg_features = emg_train.shape[2],emg_train.shape[3]
n_acc_length, n_acc_features = acc_train.shape[2],acc_train.shape[3]
n_gyr_length, n_gyr_features = gyr_train.shape[2],gyr_train.shape[3]

# Knowledge Distiller Class 

class Distiller1(tf.keras.Model):
    def __init__(self, student, teacher):
        super(Distiller1, self).__init__()
        self.teacher = teacher
        self.student = student

    def compile(
        self,
        optimizer,
        metrics,
        student_loss_fn,
        distillation_loss_fn,
        alpha=0.2,
        temperature=3,
    ):
        
        super(Distiller1, self).compile(optimizer=optimizer, metrics=metrics)
        self.student_loss_fn = student_loss_fn
        self.distillation_loss_fn = distillation_loss_fn
        self.alpha = alpha
        self.temperature = temperature

    def train_step(self, data):
        # Unpack data
        x, y = data

        # Forward pass of teacher
        teacher_predictions = self.teacher(x, training=False)

        with tf.GradientTape() as tape:
            # Forward pass of student
            student_predictions = self.student(x, training=True)

            # Compute losses
            student_loss = self.student_loss_fn(y, student_predictions)
            distillation_loss = self.distillation_loss_fn(
                tf.nn.softmax(teacher_predictions / self.temperature, axis=1),
                tf.nn.softmax(student_predictions / self.temperature, axis=1),
            )
            loss = self.alpha * student_loss + (1 - self.alpha) * distillation_loss

        # Compute gradients
        trainable_vars = self.student.trainable_variables
        gradients = tape.gradient(loss, trainable_vars)

        # Update weights
        self.optimizer.apply_gradients(zip(gradients, trainable_vars))

        # Update the metrics configured in `compile()`.
        self.compiled_metrics.update_state(y, student_predictions)

        # Return a dict of performance
        results = {m.name: m.result() for m in self.metrics}
        results.update(
            {"student_loss": student_loss, "distillation_loss": distillation_loss}
        )
        return results

    def test_step(self, data):
        # Unpack the data
        x, y = data

        # Compute predictions
        y_prediction = self.student(x, training=False)

        # Calculate the loss
        student_loss = self.student_loss_fn(y, y_prediction)

        # Update the metrics.
        self.compiled_metrics.update_state(y, y_prediction)

        # Return a dict of performance
        results = {m.name: m.result() for m in self.metrics}
        results.update({"student_loss": student_loss})
        return results

# Multi Expert Base/Teacher Model
# The Multi Expert Base/Teacher Model used in our study is a multi-channel Deep convolutional network.

# Functional Model 1 was trained on sEMG data
emg_model = Input(shape=(None,n_emg_length,n_emg_features))
emg_conv1 = TimeDistributed(Conv1D(filters=12, kernel_size=7, activation='sigmoid'), input_shape=(None,n_emg_length,n_emg_features))(emg_model)
emg_avgpool1 = TimeDistributed(AveragePooling1D(pool_size=3))(emg_conv1)
emg_conv2 = TimeDistributed(Conv1D(filters=24, kernel_size=7, activation='sigmoid'))(emg_avgpool1)
emg_avgpool2 = TimeDistributed(AveragePooling1D(pool_size=3))(emg_conv2)
emg_flat1 = TimeDistributed(Flatten())(emg_avgpool2)
emg_lstm1 = LSTM(100, dropout=0.2, recurrent_dropout=0.2, return_sequences = True)(emg_flat1)
emg_lstm2 = LSTM(100, dropout=0.2, recurrent_dropout=0.2)(emg_lstm1)
emg_batchnorm = BatchNormalization(batch_size = batch_size)(emg_lstm2)
emg_drop = Dropout(0.2)(emg_batchnorm)

# Functional Model 2 was trained on Accelerometer data
acc_model = Input(shape=(None,n_acc_length,n_acc_features)) 
acc_conv1 = TimeDistributed(Conv1D(filters=12, kernel_size=7, activation='sigmoid'), input_shape=(None,n_acc_length,n_acc_features))(acc_model)
acc_avgpool1 = TimeDistributed(AveragePooling1D(pool_size=3))(acc_conv1)
acc_conv2 = TimeDistributed(Conv1D(filters=24, kernel_size=7, activation='sigmoid'))(acc_avgpool1)
acc_avgpool2 = TimeDistributed(AveragePooling1D(pool_size=3))(acc_conv2)
acc_flat1 = TimeDistributed(Flatten())(acc_avgpool2)
acc_lstm1 = LSTM(100, dropout=0.2, recurrent_dropout=0.2, return_sequences = True)(acc_flat1)
acc_lstm2 = LSTM(100, dropout=0.2, recurrent_dropout=0.2)(acc_lstm1)
acc_batchnorm = BatchNormalization(batch_size = batch_size)(acc_lstm2)
acc_drop = Dropout(0.2)(acc_batchnorm)

# Functional Model 3 was trained on Gyroscope data
gyr_model = Input(shape=(None,n_gyr_length,n_gyr_features)) 
gyr_conv1 = TimeDistributed(Conv1D(filters=12, kernel_size=7, activation='sigmoid'), input_shape=(None,n_gyr_length,n_gyr_features))(gyr_model)
gyr_avgpool1 = TimeDistributed(AveragePooling1D(pool_size=3))(gyr_conv1)
gyr_conv2 = TimeDistributed(Conv1D(filters=24, kernel_size=7, activation='sigmoid'))(gyr_avgpool1)
gyr_avgpool2 = TimeDistributed(AveragePooling1D(pool_size=3))(gyr_conv2)
gyr_flat1 = TimeDistributed(Flatten())(gyr_avgpool2)
gyr_lstm1 = LSTM(100, dropout=0.2, recurrent_dropout=0.2, return_sequences = True)(gyr_flat1)
gyr_lstm2 = LSTM(100, dropout=0.2, recurrent_dropout=0.2)(gyr_lstm1)
gyr_batchnorm = BatchNormalization(batch_size = batch_size)(gyr_lstm2)
gyr_drop = Dropout(0.2)(gyr_batchnorm)

# All three models are concatenated
merged = concatenate([emg_drop, acc_drop, gyr_drop])
dense1 = Dense(n_outputs, activation='softmax')(merged)
model = Model(inputs=[emg_model, acc_model, gyr_model], outputs=dense1)
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# summarize
print(model.summary())

# define model
hist1 = model.fit([emg_train, acc_train, gyr_train], y_train, epochs=epochs, batch_size=batch_size, verbose=True, validation_data=([emg_test, acc_test, gyr_test], y_test)) #, epochs=epochs,callbacks=[es]
hist_arr1 = np.array([hist1.history['accuracy'],hist1.history['val_accuracy'],hist1.history['loss'],hist1.history['val_loss']])

## Model Result/Performance Plots
# Test Accuracy
plt.figure(1)
plt.plot(hist_arr1[1])
plt.ylabel('Test Accuracy (%)')
plt.xlabel('Epoch')
plt.grid(True)
plt.ylim(0.25,1.05)

# Validation Accuracy
plt.figure(2)
plt.plot(hist_arr1[0])
plt.ylabel('Train Accuracy (%)')
plt.xlabel('Epoch')
plt.grid(True)
plt.ylim(0.25,1.05)

# Test Loss
plt.figure(3)
plt.plot(hist_arr1[3])
plt.ylabel('Test Loss')
plt.xlabel('Epoch')
plt.grid(True)

# Validation Loss
plt.figure(4)
plt.plot(hist_arr1[2])
plt.ylabel('Train Loss')
plt.xlabel('Epoch')
plt.grid(True)

# Multi Expert Student Model
# The Multi Expert Student Model used in our study is a multi-channel Deep convolutional network.

# Functional Model 1 was trained on sEMG data
stu_emg_model = Input(shape=(None,n_emg_length,n_emg_features))
stu_emg_conv1 = TimeDistributed(Conv1D(filters=2, kernel_size=4, activation='sigmoid'), input_shape=(None,n_emg_length,n_emg_features))(stu_emg_model)
stu_emg_avgpool1 = TimeDistributed(AveragePooling1D(pool_size=2))(stu_emg_conv1)
stu_emg_conv2 = TimeDistributed(Conv1D(filters=4, kernel_size=4, activation='sigmoid'))(stu_emg_avgpool1)
stu_emg_avgpool2 = TimeDistributed(AveragePooling1D(pool_size=2))(stu_emg_conv2)
stu_emg_flat1 = TimeDistributed(Flatten())(stu_emg_avgpool2)
stu_emg_lstm1 = LSTM(50, dropout=0.2, recurrent_dropout=0.2, return_sequences = True)(stu_emg_flat1)
stu_emg_lstm2 = LSTM(50, dropout=0.2, recurrent_dropout=0.2)(stu_emg_lstm1)
stu_emg_batchnorm = BatchNormalization(batch_size = batch_size)(stu_emg_lstm2)
stu_emg_drop = Dropout(0.2)(stu_emg_batchnorm)

# Functional Model 2 was trained on Accelerometer data
stu_acc_model = Input(shape=(None,n_acc_length,n_acc_features)) 
stu_acc_conv1 = TimeDistributed(Conv1D(filters=2, kernel_size=4, activation='sigmoid'), input_shape=(None,n_acc_length,n_acc_features))(stu_acc_model)
stu_acc_avgpool1 = TimeDistributed(AveragePooling1D(pool_size=2))(stu_acc_conv1)
stu_acc_conv2 = TimeDistributed(Conv1D(filters=4, kernel_size=4, activation='sigmoid'))(stu_acc_avgpool1)
stu_acc_avgpool2 = TimeDistributed(AveragePooling1D(pool_size=2))(stu_acc_conv2)
stu_acc_flat1 = TimeDistributed(Flatten())(stu_acc_avgpool2)
stu_acc_lstm1 = LSTM(50, dropout=0.2, recurrent_dropout=0.2, return_sequences = True)(stu_acc_flat1)
stu_acc_lstm2 = LSTM(50, dropout=0.2, recurrent_dropout=0.2)(stu_acc_lstm1)
stu_acc_batchnorm = BatchNormalization(batch_size = batch_size)(stu_acc_lstm2)
stu_acc_drop = Dropout(0.2)(stu_acc_batchnorm)

# Functional Model 3 was trained on Gyroscope data
stu_gyr_model = Input(shape=(None,n_gyr_length,n_gyr_features)) 
stu_gyr_conv1 = TimeDistributed(Conv1D(filters=2, kernel_size=4, activation='sigmoid'), input_shape=(None,n_gyr_length,n_gyr_features))(stu_gyr_model)
stu_gyr_avgpool1 = TimeDistributed(AveragePooling1D(pool_size=2))(stu_gyr_conv1)
stu_gyr_conv2 = TimeDistributed(Conv1D(filters=4, kernel_size=4, activation='sigmoid'))(stu_gyr_avgpool1)
stu_gyr_avgpool2 = TimeDistributed(AveragePooling1D(pool_size=2))(stu_gyr_conv2)
stu_gyr_flat1 = TimeDistributed(Flatten())(stu_gyr_avgpool2)
stu_gyr_lstm1 = LSTM(50, dropout=0.2, recurrent_dropout=0.2, return_sequences = True)(stu_gyr_flat1)
stu_gyr_lstm2 = LSTM(50, dropout=0.2, recurrent_dropout=0.2)(stu_gyr_lstm1)
stu_gyr_batchnorm = BatchNormalization(batch_size = batch_size)(stu_gyr_lstm2)
stu_gyr_drop = Dropout(0.2)(stu_gyr_batchnorm)

# All three models are concatenated
stu_merged = concatenate([stu_emg_drop, stu_acc_drop, stu_gyr_drop])
stu_dense1 = Dense(n_outputs, activation='softmax')(stu_merged)
stu_model = Model(inputs=[stu_emg_model, stu_acc_model, stu_gyr_model], outputs=stu_dense1)
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
# summarize
print(stu_model.summary())

#Distill Student to Teacher

distiller = Distiller1(student=stu_model, teacher=model) #invokes the KD class
distiller.compile(
    optimizer=adam,
    metrics=['accuracy'],
    student_loss_fn=keras.losses.CategoricalCrossentropy(from_logits=True),
    distillation_loss_fn=keras.losses.KLDivergence(),
    alpha=0.1,
    temperature=5,
    )
hist2= distiller.fit([emg_train, acc_train, gyr_train], y_train, epochs=50, batch_size=batch_size, verbose=True, validation_data=([emg_test, acc_test, gyr_test], y_test)) #, epochs=epochs,callbacks=[es]
hist_arr2 = np.array([hist2.history['accuracy'],hist2.history['student_loss'],hist2.history['distillation_loss'],hist2.history['val_accuracy'],hist2.history['val_student_loss']])

# Test Accuracy
plt.figure(5)
plt.plot(hist_arr2[0])
plt.ylabel('Test Accuracy (%)')
plt.xlabel('Epoch')
plt.grid(True)

# Validation Accuracy
plt.figure(2)
plt.plot(hist_arr2[3])
plt.ylabel('Train Accuracy (%)')
plt.xlabel('Epoch')
plt.grid(True)

# Test Loss
plt.figure(3)
plt.plot(hist_arr2[1])
plt.ylabel('Test Loss')
plt.xlabel('Epoch')
plt.grid(True)

# Validation Loss
plt.figure(4)
plt.plot(hist_arr2[4])
plt.ylabel('Train Loss')
plt.xlabel('Epoch')
plt.grid(True)

# Distillation Loss
plt.figure(4)
plt.plot(hist_arr2[2])
plt.ylabel('Teacher over Student Distillation Loss')
plt.xlabel('Epoch')
plt.grid(True)

# Base Model Accuracy
y1_pred = model.evaluate([emg_test, acc_test, gyr_test], y_test, verbose=0)[1]
print("Base Model Accuracy:","{:.3f}%".format(y1_pred*100))

# Distilled Model Accuracy
y2_pred = distiller.evaluate([emg_test, acc_test, gyr_test], y_test, verbose=0)[1]
print("Distilled Model Accuracy","{:.3f}%".format(y2_pred*100))


# Set base model type to .h5 and save to acquire size
import tempfile
import os
_, keras_file = tempfile.mkstemp('.h5')
tf.keras.models.save_model(model, keras_file, include_optimizer=False)
print('Saved basel model to:', keras_file)

# Set student model type to .h5 and save to acquire size
import tempfile
import os
_, KD_keras_file = tempfile.mkstemp('.h5')
tf.keras.models.save_model(stu_model, KD_keras_file, include_optimizer=False)
print('Saved Distilled Keras model to:', KD_keras_file)

def get_gzipped_model_size(file):
  # Returns size of gzipped model, in bytes.
  import os
  import zipfile

  _, zipped_file = tempfile.mkstemp('.zip')
  with zipfile.ZipFile(zipped_file, 'w', compression=zipfile.ZIP_DEFLATED) as f:
    f.write(file)

  return os.path.getsize(zipped_file)

  print("Size of Base Model: %.2f bytes" % (get_gzipped_model_size(keras_file)))

  print("Size of Distilled Model: %.2f bytes" % (get_gzipped_model_size(KD_keras_file)))

# Bar graph for size comparison
data = {'Size of Base Mode':get_gzipped_model_size(keras_file)} 
data1 = {'Size of Distilled Model':get_gzipped_model_size(KD_keras_file)}
modell = list(data.keys())
size = list(data.values())
modelll = list(data1.keys())
sizee = list(data1.values())

sizees = [get_gzipped_model_size(keras_file),get_gzipped_model_size(KD_keras_file)]
sizeees = [get_gzipped_model_size(KD_keras_file),get_gzipped_model_size(keras_file)]
  
fig = plt.figure(figsize = (10, 5))
for x, y, p in zip(modell, size, sizees):
   plt.text(x, y, p)

for x1, y1, p1 in zip(modelll, sizee,sizeees):
   plt.text(x1, y1, p1)
# creating the bar plot
plt.bar(modell, size, color ='maroon',
        width = 0.4)
plt.bar(modelll,sizee, color = 'pink', width = 0.4)

plt.ylabel("Size of Model in MB")
plt.xlabel("Model Name")
plt.title("Size difference through Knowledge Distillation")
plt.show()