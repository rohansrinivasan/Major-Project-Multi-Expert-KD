# Major Project Work on Multi Expert Knowledge Distillation
Hi! This repository includes my project work done during my 8th semester major project "Practical Sign Language Recognition System using Deep Learning."
This is a continuation of my work in my minor project and is a primary focus on the deep learning model compression technique knowledge distillation.

## What is model compression?
It is essentially a method of deploying high level deep learning models in devices with low specifications and power without hindering model performance. Model compression helps in increasing the lifetime of a deep learning model, by reducing its complexity and size so it can be easily implemented in several devices, making it more and more viable for years to come.
Different types of model compression techniques are - 
* Pruning: A method where connections between neurons or sometimes entire neurons, channels or filters are removed from trained models.
* Quantization: Achieves model compression by reducing the size of weights present.
* Low-Rank Approximation: It is a matrix decomposition technique done to evaluate useful parameters within a network. Similar to pruning in a way.
* Selective Attention: Only the layers and objects of interest and those which contribute towards the mode are considered, and rest of the elements are discarded.
* Knowledge Distillation: Knowledge distillation is built on the idea that training and inference are two different tasks, and that the same model need not be used for each. (This is the technique used in the project)

## What is Knowledge Distillation?
Knowledge distillation is built on the idea that training and inference are two different tasks, and that the same model need not be used for each. Here a base expert model called the teacher model is trained, and its learning is then used to teach a much smaller student model with fewer parameters in comparison to mimic the base model’s performance. The goal is so that smaller student model has the same distribution as the larger teacher model and thus can be used for deployment.

## What is Multi-Expert Knowledge Distillation? and why it is implemented in this project! 
Multi-Teacher Knowledge Distillation is an extension to Knowledge Distillation, where instead of one teacher distributing over a student, we use multiple teacher’s as an ensemble or average their weights and use their data to distribute over a student model. As mentioned before, we know that Knowledge Distillation is an effective model compression technique, but it has its limits. The knowledge from a single teacher may be limited to some depth and also can be biased which results in a low-quality student model. Using a multi-teacher base model severely reduces the overfitting bias in comparison to a single teacher base model and transfers more generalized knowledge to the student modeland an ensemble of teachers distilling over a student provides much more promising performance from the student model as compared to a single one. Thus, it is quite clear that using multiple teacher models to train a single student model is much more viable, future-proof and boasts better accuracy than compared to a single teacher model. This is the motivation for Multi – Teacher Knowledge Distillation.

## Dependencies
* Python (3.6 or higher)
* Pandas
* Keras 
* Tensorflow – Python library for developing deep neural networks.
* Numpy – Python library used for working with arrays.
* Matplotlib – It is a Python library used for plotting graphs to visualise data.
* Scipy – Python library used for solving mathematical, technical and scientific
problems.
* Scikit-Learn – Python library for machine learning. It contains various machine
learning algorithms within it.
* Seaborn – A Python library built on top of Matplotlib. It is also used to visualise
data.

### This project is ran/tested on Google Colab. 

# Dataset 
* The dataset used in this study was acquired with the help of 5 volunteers. Different modalities were captured using Surface Electro-myogram or sEMG, tri-axis gyroscope and tri-axis accelerometer, and a multi-modal and multi-sensor database of signals for 50 commonly used signs from the Indian sign language (ISL), both static and dynamic. There are a total of 5,000 observations (50 signs, performed by 5 subjects, each sign performed 20 times by each subject), therefore making it 100 observations for each sign.
* Signals were recorded using six Delsys wireless sensors, consisting of one sEMG sensor and one IMU containing a tri-axis accelerometer and a tri-axis gyroscope each. The sampling period sEMG sensors was 900 μs and for accelerometer and gyroscopes, 6.7 ms and 16 bits per sample.

# Model Architecture

## Multi-Teacher Model Parameters (Architecture and Parameter of each functional model remains the same)
* 1D convolutional layer with 12 filters after the initial input layer (total 26 filters) with Sigmoid activation function.
* Average-pooling layer, with the pool size as 3
* Another 1D convolutional layer with 24 filters after the initial input layer (total 72 filters) with Sigmoid activation function.
* Another average pool layer with the pool size as 3.
* Flatten layer which converts the data into one-dimension to make it appropriate for the next layer.
* 2 units of LSTM layer (Long Short-Term Memory), having 100 in each unit. (200 per model, 600 for the entire expert model). These networks are a similar to an RNN, and are capable of learning order dependencies. Such a layer does data processing, while passing on information as it moves forward.
* A Batch normalization layer, with the batch size as 64.
* A Dropout layer with 20% dropout to avoid overfitting.
* A Dense or a fully connected layer with 50 neurons owing to the 50 target classes
### Trainable Parameters: 693,170


## Multi-Student Model Parameters (Architecture and Parameter of each functional model remains the same)
* 1D convolutional layer with 2 filters after the initial input layer (total 6 filters) with Sigmoid activation function.
* Average-pooling layer, with the pool size as 3
* Another 1D convolutional layer with 4 filters after the initial input layer (total 12 filters) with Sigmoid activation function.
* Another average pool layer with the pool size as 3.
* Flatten layer which converts the data into one-dimension to make it appropriate for the next layer.
* 2 units of LSTM layer (Long Short-Term Memory), having 50 in each unit. (100 per model, 300 for the entire expert model). These networks are a similar to an RNN, and are capable of learning order dependencies. Such a layer does data processing, while passing on information as it moves forward.
* A Batch normalization layer, with the batch size as 64.
* A Dropout layer with 20% dropout to avoid overfitting.
* A Dense or a fully connected layer with 50 neurons owing to the 50 target classes
### Trainable Parameters: 168,432


# Results
### Multi-Teacher Model Accuracy: 93.600%  
### Multi-Teacher Model Size: 2,579,392 bytes

### Multi-Student Model Accuracy: 91.485%  
### Multi-Student Model Size: 633,673 bytes
