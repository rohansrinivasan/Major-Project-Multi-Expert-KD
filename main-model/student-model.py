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
