import numpy as np
import io
import os
import cv2
from keras.applications.xception import Xception, preprocess_input
from keras.layers import Dropout, GlobalAveragePooling2D, Dense
from keras.models import Sequential
from flask import Flask, jsonify, request, render_template
from keras.preprocessing import image
from PIL import Image
from werkzeug import secure_filename

app = Flask(__name__) # create a Flask app

@app.route('/')
def predictbreed():
    return render_template('DogBreedsPredictionApp.html')

# helper
def path_to_tensor(img_path, target_size=(224, 224)):
    # loads RGB image as PIL.Image.Image type
    img = image.load_img(img_path, target_size=target_size)
    # convert PIL.Image.Image type to 3D tensor with shape (224, 224, 3)
    x = image.img_to_array(img)
    # convert 3D tensor to 4D tensor with shape (1, 224, 224, 3) and return 4D tensor
    return np.expand_dims(x, axis=0)

def extract_Xception(tensor):
	from keras.applications.xception import Xception, preprocess_input
	return Xception(weights='imagenet', include_top=False).predict(preprocess_input(tensor))

def get_Xception_output_shape():
    model = Xception(weights='imagenet', include_top=False)
    return model.output_shape

def get_nnet():
    output_shape = get_Xception_output_shape()
    Xception_model = Sequential()
    Xception_model.add(GlobalAveragePooling2D(input_shape=output_shape[1:]))
    Xception_model.add(Dense(133, activation='softmax'))

    Xception_model.load_weights('saved_models/weights.best.DogXception.hdf5')

    return Xception_model

def getBreedsLabels(breed_index):
    dog_breeds = np.load('dog_breed_labels.npz')
    return dog_breeds['breed_labels'][breed_index][0]

def Xception_predict_breed(img_path):
    # extract bottleneck features
    Xception_model = get_nnet()
    bottleneck_feature = extract_Xception(path_to_tensor(img_path))
    # obtain predicted vector
    predicted_vector = Xception_model.predict(bottleneck_feature)
    # return dog breed that is predicted by the model
    return getBreedsLabels([np.argmax(predicted_vector)])

def dog_detector(img_path):
    base_model = Xception(weights='imagenet')
    img = preprocess_input(path_to_tensor(img_path, target_size=(299, 299)))
    prediction = np.argmax(base_model.predict(img))
    return ((prediction <= 268) & (prediction >= 151))

def face_detector(img_path):
    face_cascade = cv2.CascadeClassifier('haarcascades/haarcascade_frontalface_alt.xml')
    img = cv2.imread(img_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray)
    return len(faces) > 0

@app.route('/predict', methods=['POST', 'GET'])
def predict():
    file = request.files['file']
    extension = os.path.splitext(file.filename)[1]
    fileName = os.path.splitext(file.filename)[0]
    f_name = fileName + extension
    image_path = os.path.join("static\\Images_Test", f_name)
    file.save(image_path)

    dogDetected = dog_detector(image_path)

    if dogDetected:
        predicted_dog_breed = Xception_predict_breed(image_path)
        prediction_result = 'Hello dog, Your predicted breed is ' + predicted_dog_breed
    elif face_detector(image_path):
        predicted_dog_breed = Xception_predict_breed(image_path)
        prediction_result = 'Hello human, You look like a ' + predicted_dog_breed
    else:
        predicted_dog_breed = prediction_result = 'No dogs or human faces resemble dogs!'

    return jsonify({'prediction': predicted_dog_breed,
                    'predictionResult': prediction_result,
                    'imagePath': image_path})

if __name__ == '__main__':
    # this will start a local server
    app.run(debug=True)