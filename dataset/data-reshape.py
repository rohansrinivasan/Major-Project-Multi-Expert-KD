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
