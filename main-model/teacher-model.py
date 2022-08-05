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