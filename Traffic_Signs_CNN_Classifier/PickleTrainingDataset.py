import numpy as np
from PIL import Image
import pickle
import os
import pandas as pd

test_labels_df = pd.read_csv('./original-traffic-signs-dataset/GT-final_test.csv', sep=';')

test_file_names = test_labels_df.values[:, 0]
test_class_labels = test_labels_df.values[:, -1]

train_data = {}
train_data['features'] = np.ndarray(shape=(39209, 32, 32, 3), dtype=np.uint8)
train_data['labels'] = np.ndarray(shape=(39209,), dtype=np.uint8)

test_data = {}
test_data['features'] = np.ndarray(shape=(12630, 32, 32, 3), dtype=np.uint8)
test_data['labels'] = np.ndarray(shape=(12630,), dtype=np.uint8)

train_dir = './original-traffic-signs-dataset/GTSRB_Final_Training_Images/GTSRB/Final_Training/Images'
test_dir = './original-traffic-signs-dataset/GTSRB_Final_Test_Images/GTSRB/Final_Test/Images'

index = 0

for subdir, dirs, files in os.walk(train_dir):
    subdir_name = subdir.split('\\')[-1]
    for file in files:
        filepath = os.path.join(subdir, file)
        if filepath.endswith('.ppm'):
            img = Image.open(filepath).convert('RGB')
            img = img.resize((32, 32), Image.ANTIALIAS)
            train_data['features'][index] = np.asarray(img)
            train_data['labels'][index] = int(subdir_name)
            index = index + 1

print("Dump to train.pkl...")
with open("./traffic-signs-dataset/train.pkl", 'wb') as pfile:
    pickle.dump(train_data, pfile, protocol=pickle.HIGHEST_PROTOCOL)

for subdir, dirs, files in os.walk(test_dir):
    for index, file in enumerate(files):
        filepath = os.path.join(subdir, file)
        if filepath.endswith('.ppm'):
            classId = test_labels_df[test_labels_df['Filename']==file]['ClassId']
            print(file, int(classId), index)
            img = Image.open(filepath).convert('RGB')
            img = img.resize((32, 32), Image.ANTIALIAS)
            test_data['features'][index] = np.asarray(img)
            test_data['labels'][index] = int(classId)

print("Dump to test.pkl...")
with open("./traffic-signs-dataset/test.pkl", 'wb') as pfile:
    pickle.dump(test_data, pfile, protocol=pickle.HIGHEST_PROTOCOL)