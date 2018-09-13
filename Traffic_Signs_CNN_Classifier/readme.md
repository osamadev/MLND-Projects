I. Definition
Project Overview
The problem domain this project is trying to tackle is to recognize and classify the traffic signs in the real-world to help self-driving cars (as a typical use case) to do this task efficiently while driving on the roads. The dataset used in that project is German Traffic Signs Dataset (1). This dataset has more than 50,000 images in total classified into 43 classes. Due to the ambitious projects to build self-driving cars, classification of traffic signs became very essential task to achieve this.
Generally, the humans are capable to recognize the traffic signs with very high level of accuracy (humans can do this task with accuracy of 98.81%) and this could be considered as a real challenge to the implemented solution which also has to achieve a high level of accuracy and correctness near to the humans’ benchmark. 
Problem Statement
Classification of the Traffic signs is very essential problem that has to be tackled very well in any self-driving car solution. As we expect human drivers to be highly capable to recognize the traffic signs, we expect also that the auto-drivers (self-driving cars) to be the same to achieve the safety aspects while driving on the roads. Traffic signs recognition is a multi-class classification problem with unbalanced class frequencies. There is a wide range of variations among the different classes of the traffic signs in terms of shapes, colors, having text inside the traffic signal or not. On contrary, parts of these classes are very similar to each other such as the speed limit signs.
In reality, there are large variations in the visual appearance of the traffic signs due to the weather conditions, illumination changes, rotations, the time of the day (night time or day time)…etc. All of these factors should be considered when we design a solution for this challenging real-world problem.
The implemented solution here to solve this problem is mainly building a convolutional neural network classifier (CNN classifier) of multi-convolution layers and max-pooling layers, then compile and train that model using the inputs from the training dataset. This CNN classifier has been trained from scratch to able to detect the patterns in the images and to learn the low and high level features of the images given in the training dataset. Finally this trained classifier has been evaluated against the test dataset to assess its accuracy and ensure it generalizes well for the unseen instances in the real-world.
Metrics
The evaluation metrics that I have used to evaluate both the solution model and the benchmark model is the test accuracy. Model’s accuracy is the number of instances which are correctly classified divided by the total number of test instances in our test dataset. The accuracy is represented by the following formula:
Model’s Accuracy=(No.of correctly classified instances   )/(Total number of test instances )
Also I have used precision and recall metrics; here are the formulas of Precision and Recall subsequently:
Precision=(True Positives   )/(True Positives+False Positives )
Recall=(True Positives   )/(True Positives+False Negatives )
Basically, this is a classification problem that is why I have chosen the above evaluation metrics which are suitable for that type of problems.
Using the final results of the solution model and the predicted class labels of the test dataset samples, I also plotted the confusion matrix to show the predicted labels versus the true or actual class labels. 

II. Analysis
Data Exploration
The dataset that I used for this project is the German Traffic Signs Dataset (1). This dataset has more than 50,000 images in total classified into 43 classes. The training data set contains 39,209 training images in 43 classes and the test dataset contains 12,630 test images. The original image format is “PPM (Portable Pixmap, P6)” which is basically RGB format. The images are not necessary squared as image sizes vary between 15x15 to 250x250 pixels. The images in the training dataset are grouped in 43 directories (folders); the name of each folder is the encoding of the corresponding class label in a format of numbers from 0 to 42.
The images contain one traffic sign each and Physical traffic sign instances are unique within the dataset which means each real-world traffic sign only occurs once. The actual traffic sign is not necessarily centered within the image and there is about 10% border around the actual traffic sign. This dataset comes with Annotations provided in CSV files. Fields in these CSV files are separated by ";"   (semicolon). Annotations contain the following information:
	Class Id: Assigned class label (For the training dataset)
	Filename: Filename of corresponding image
	Width: Width of the image
	Height: Height of the image
	ROI.x1: X-coordinate of top-left corner of traffic sign bounding box
	ROI.y1: Y-coordinate of top-left corner of traffic sign bounding box
	ROI.x2: X-coordinate of bottom-right corner of traffic sign bounding box
	ROI.y2: Y-coordinate of bottom-right corner of traffic sign bounding box

As initial preprocessing of the original dataset, we iterated over the images of the training and test datasets to resize each image to a fixed width and height (32 x 32) pixels, and then convert each resized RGB image to a numpy 2-D array with numbers range from 0 to 255 which represents the intensity of each pixel. After that initial preprocessing, I dumped this data into two pickle files; one for the training data “Train.pkl” and the other one for test data “Test.pkl”, to save the initial transformations that we have done and make the further processing easier. 
Note: I have uploaded the two pickle files on OneDrive and they can be reached and downloaded through this link https://1drv.ms/f/s!Apt9CJrW-9NSghXkbEbbY50ZAe-H
The training dataset has been split into two splits; one split for the actual training dataset which represents 88% of the original training data and the other split for the validation dataset which represents the other 12%.
I already did analysis over the data and here a list that summarizes the statistics about the samples count per each class label in the actual training dataset after splitting original data into training and validation datasets.

ClassId  class Label                                   Samples Count
0        Speed limit (20km/h)                           	188
1        Speed limit (30km/h)                           	1905
2        Speed limit (50km/h)                           	1991
3        Speed limit (60km/h)                           	1235
4        Speed limit (70km/h)                           	1750
5        Speed limit (80km/h)                                1659
6        End of speed limit (80km/h)                         382
7        Speed limit (100km/h)                               1289
8        Speed limit (120km/h)                               1216
9        No passing                                          1309
10       No passing for vehicles over 3.5 metric tons        1792
11       Right-of-way at the next intersection               1172
12       Priority road                                       1827
13       Yield                                               1879
14       Stop                                                678
15       No vehicles                                         556
16       Vehicles over 3.5 metric tons prohibited            363
17       No entry                                            972
18       General caution                                     1072
19       Dangerous curve to the left                         188
20       Dangerous curve to the right                        318
21       Double curve                                        291
22       Bumpy road                                          341
23       Slippery road                                       440
24       Road narrows on the right                           239
25       Road work                                           1327
26       Traffic signals                                     523
27       Pedestrians                                         214
28       Children crossing                                   479
29       Bicycles crossing                                   232
30       Beware of ice/snow                                  395
31       Wild animals crossing                               680
32       End of all speed and passing limits                 218
33       Turn right ahead                                    599
34       Turn left ahead                                     367
35       Ahead only                                          1058
36       Go straight or right                                339
37       Go straight or left                                 187
38       Keep right                                          1828
39       Keep left                                           262
40       Roundabout mandatory                                322
41       End of no passing                                   209
42       End of no passing by vehicles over 3.5 metric tons  212



- The implementation of the CNN classifier is available in "Traffic_Signs_Classifier.ipynb" file.
 
- I converted the original dataset that has PPM files to a pickle files, one for training dataset and the other one for the test dataset. 

 - To check the code that I implemented to convert the original dataset to pickle files , please check the "PickleTrainingDataset.py" file which is already attached with the submission.

 - To minimize the size of the capstone submission, I have uploaded the two pickle files (train.pkl & test.pkl) to OneDrive and it could be reached through the following link: https://1drv.ms/f/s!Apt9CJrW-9NSghXkbEbbY50ZAe-H

- I saved the class labels of the traffic signs in npz file format called "traffic_sign_labels.npz"

- The Flask app of the solution model is in "TrafficSigns_WebApp" folder.

- The weights of the solution model is available in "saved_models" folder. This folder includes two main files as follows:
	- "weights.best.model.cv.hdf5" which has the weights before data augmentation.
	- "weights.best.model.optimized.hdf5" which has the weights after applying the data augmentation.


- The train and test datasets in a pickle format should be in "traffic-signs-dataset" folder as shown in the Jupyter notebook.



