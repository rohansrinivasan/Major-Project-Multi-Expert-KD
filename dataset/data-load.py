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