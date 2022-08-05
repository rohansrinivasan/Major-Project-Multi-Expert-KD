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
![1_8KqNtABnNXM527JK9UuBUQ](https://user-images.githubusercontent.com/102278418/183052478-bd426308-9eed-4a9b-9bef-10227f193927.jpeg)

Knowledge distillation is built on the idea that training and inference are two different tasks, and that the same model need not be used for each. Here a base expert model called the teacher model is trained, and its learning is then used to teach a much smaller student model with fewer parameters in comparison to mimic the base model’s performance. The goal is so that smaller student model has the same distribution as the larger teacher model and thus can be used for deployment.

## What is Multi-Expert Knowledge Distillation? and Why it is implemented in this project! 
![The-generic-framework-for-multi-teacher-distillation](https://user-images.githubusercontent.com/102278418/183052443-dd5f961b-abb0-4a3c-9a53-87ba5309a953.png)

Multi-Teacher Knowledge Distillation is an extension to Knowledge Distillation, where instead of one teacher distributing over a student, we use multiple teacher’s as an ensemble or average their weights and use their data to distribute over a student model. As mentioned before, we know that Knowledge Distillation is an effective model compression technique, but it has its limits. The knowledge from a single teacher may be limited to some depth and also can be biased which results in a low-quality student model. Using a multi-teacher base model severely reduces the overfitting bias in comparison to a single teacher base model and transfers more generalized knowledge to the student modeland an ensemble of teachers distilling over a student provides much more promising performance from the student model as compared to a single one. Thus, it is quite clear that using multiple teacher models to train a single student model is much more viable, future-proof and boasts better accuracy than compared to a single teacher model. This is the motivation for Multi – Teacher Knowledge Distillation.

## Dependencies
* Python (3.6 or higher)

* Pandas
* Keras 
* Tensorflow 
* Numpy 
* Matplotlib 
* Scipy 
* Scikit-Learn 
* Seaborn 

### This project is ran/tested on Google Colab. 

# Dataset 
* The dataset used in this study was acquired with the help of 5 volunteers. Different modalities were captured using Surface Electro-myogram or sEMG, tri-axis gyroscope and tri-axis accelerometer, and a multi-modal and multi-sensor database of signals for 50 commonly used signs from the Indian sign language (ISL), both static and dynamic. There are a total of 5,000 observations (50 signs, performed by 5 subjects, each sign performed 20 times by each subject), therefore making it 100 observations for each sign.
* Signals were recorded using six Delsys wireless sensors, consisting of one sEMG sensor and one IMU containing a tri-axis accelerometer and a tri-axis gyroscope each. The sampling period sEMG sensors was 900 μs and for accelerometer and gyroscopes, 6.7 ms and 16 bits per sample.

# Model Architecture
![base model architecture](https://user-images.githubusercontent.com/102278418/183045777-7aac6470-e340-4b3d-9c12-5595e0503a4a.png)
![Screen Shot 2022-04-25 at 12 07 10 AM](https://user-images.githubusercontent.com/102278418/183052209-f1592747-ff31-4fde-bb96-c97437f64f13.png)
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
### Epochs Trained on : 100


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
### Epochs Trained on : 50

# Results
## Multi-Teacher Model 
Accuracy: 93.600%  
Model Size: 2,579,392 bytes

![teach_train_acc](https://user-images.githubusercontent.com/102278418/183049716-b51d9478-5dd9-4db5-89a0-24c367c62f4a.jpg)
![teach_train_loss](https://user-images.githubusercontent.com/102278418/183049786-744fc441-b362-4d3e-87bb-243012ce9782.jpg)
![teach_test_acc](https://user-images.githubusercontent.com/102278418/183047675-00d9ade3-d2c7-48d9-8b1f-c1c1520b94b4.jpg)
![teach_test_loss](https://user-images.githubusercontent.com/102278418/183049869-b387694b-9c92-4280-940a-37a8b97e223a.jpg)

## Multi-Student Model 
Accuracy: 91.485%  
Model Size: 633,673 bytes

![distillation_test_accuracy](https://user-images.githubusercontent.com/102278418/183050108-66dcbcb1-2632-4bf8-af8e-93228bc4f4a6.jpg)
![distillation Loss](https://user-images.githubusercontent.com/102278418/183050268-0b11fb5b-7b78-41b8-a239-938caae7446d.jpg)

### Model Size Comparison
![model size bar](https://user-images.githubusercontent.com/102278418/183050629-d7221d23-e451-4093-9ebd-a28bd3e70bd3.jpg)

# Acknowledgement
We would like to recognize the funding support provided by the Science & Engineering Research Board, a statutory body of the Department of Science & Technology (DST), Government of India, SERB file number ECR/2016/000637.

# Team Members
* Rohan Srinivasan (Me) (Linkedin: https://www.linkedin.com/in/rohan-srinivasan-2457591b1/)
* Sanjana Golaya (Linkedin: https://www.linkedin.com/in/sanjana-golaya/)

# Faculty Guide
* Dr. Rinki Gupta (Linkedin: https://www.linkedin.com/in/rinki-gupta-019666133/)

# References
* https://machinelearningmastery.com/develop-n-gram-multichannel-convolutional-neural-network-sentiment-analysis/
* https://machinelearningmastery.com/how-to-develop-convolutional-neural-network-models-for-time-series-forecasting/
* KD: https://keras.io/examples/vision/knowledge_distillation/
* Ensemble KD: https://machinelearningmastery.com/weighted-average-ensemble-for-deep-learning-neural-networks/

