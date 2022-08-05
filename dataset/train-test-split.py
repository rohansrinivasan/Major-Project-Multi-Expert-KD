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

#%%  multiple input model
n_length, n_features= X_train.shape[2],X_train.shape[3]
n_emg_length, n_emg_features = emg_train.shape[2],emg_train.shape[3]
n_acc_length, n_acc_features = acc_train.shape[2],acc_train.shape[3]
n_gyr_length, n_gyr_features = gyr_train.shape[2],gyr_train.shape[3]