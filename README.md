Collection of machine learning and deep learning projects that I have successfully finished during my Machine Learning Engineer Nano-degree Program from Udacity.

## Projects Summary:
**Here is a quick summary of the projects listed under this collection:**

### Creating Customer Segments for a Wholesale Distributor using Clustering and PCA Techniques (Unsupervised Learning)

Reviewed unstructured data to understand the patterns and natural categories that the data fits into. Used multiple algorithms (K-Means and GMM) both empirically and theoretically, then compared and contrasted their results. Made predictions about the natural categories of multiple types in a dataset, then checked these predictions against the result of unsupervised analysis.
For dimensionality reduction, I applied the principal components analysis (PCA) to define the most important features that drive the trend in the dataset (have the maximum variance). 
For clustering the dataset, I have used Gaussian Mixture Model (Expectation Maximization algorithm).


### Train a Smart Cab Agent to Drive Safely and Efficiently Using Q-Learning

Applied reinforcement learning to build a simulated vehicle navigation agent. This project involved modeling a complex control problem in terms of limited available inputs, and designing a scheme to automatically learn an optimal driving strategy based on rewards and penalties.	


### Dog Breed Classifier using CNN and Transfer Learning

This project is about building a dog breed classifier using Convolution Neural Network (CNN) and Transfer Learning. The test accuracy of this CNN model to predict the dog breeds from its images is about 87%. In the final model using transfer learning, I have used Xception pre-trained model to act as a features extractor of the dogs’ dataset, and then apply fully connected dense layers to classify the dog images into its predicted breed. Finally I converted the final model into Flask Web App to be production-ready solution. The dataset has a limited number of instances, so to protect against overfitting and augment the size of training dataset; we used data augmentation technique to increase the number of the instances in the training dataset as well as the validation dataset.


### Classification of Traffic Signs in the Wild Using CNN

This project is to recognize the traffic signs in the wild (real-world) which is one of the main tasks for any self-driving car project. It is a computer vision classification problem that I’ve tackled by building a CNN model trained from scratch to do the job. The original dataset has more than 50,000 traffic sign images with different sizes collected from the real-world, these images are categorized into 43 target labels. A cross validation technique using randomized grid search has been applied to find-tune the hyperparameters of the model. There are large variations in the visual appearance of the traffic signs in this dataset due to the weather conditions, illumination changes, rotations, the time of the day (night time or day time)…etc. and this was a challenge. My final solution model was able to achieve about 98.5% prediction accuracy on the test dataset which has 12,630 instances. I also converted my final solution model into a REST APIs to consume it later in a mobile or a web App.


### Finding Donors for Charity (Supervised Learning)

Investigated factors that affect the likelihood of charity donations being made based on real census data. Developed a naive classifier to compare testing results to. Trained and tested several supervised machine learning models on preprocessed census data to predict the likelihood of donations. Selected the best model based on accuracy, a modified F-scoring metric, and algorithm efficiency. The final model was based on Gradient Boosting Classifier (Ensemble Technique) and this model achieved more than 87% test score accuracy.

