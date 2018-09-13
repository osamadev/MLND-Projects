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



