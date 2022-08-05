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