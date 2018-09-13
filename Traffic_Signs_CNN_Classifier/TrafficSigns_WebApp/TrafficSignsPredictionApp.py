import numpy as np
import io
import os
import cv2
from keras.applications.xception import Xception, preprocess_input
from keras.layers import Dropout, Dense, Conv2D, Flatten, BatchNormalization, MaxPooling2D
from keras.models import Sequential
from flask import Flask, jsonify, request, render_template
from keras.preprocessing import image
from PIL import Image
from werkzeug import secure_filename

app = Flask(__name__) # create a Flask app

@app.route('/')
def load_home_page():
    return render_template('TrafficSignsPredictionApp.html')

# helper
def path_to_tensor(img_path, target_size=(32, 32)):
    # loads RGB image as PIL.Image.Image type
    img = image.load_img(img_path, target_size=target_size)
    # convert PIL.Image.Image type to 3D tensor with shape (32, 32, 3)
    x = image.img_to_array(img)
    # convert 3D tensor to 4D tensor with shape (1, 224, 224, 3) and return 4D tensor
    return np.expand_dims(x, axis=0)

def create_model():
    model = Sequential()
    model.add(
        Conv2D(filters=32, kernel_size=(3, 3), strides=1, activation='relu', padding='same', input_shape=(32, 32, 3)))
    model.add(BatchNormalization())
    model.add(Conv2D(filters=32, kernel_size=(3, 3), strides=1, activation='relu', padding='same'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=2))

    model.add(Conv2D(filters=64, kernel_size=(3, 3), strides=1, activation='relu', padding='same'))
    model.add(BatchNormalization())
    model.add(Conv2D(filters=64, kernel_size=(3, 3), strides=1, activation='relu', padding='same'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=2))

    model.add(Conv2D(filters=128, kernel_size=(3, 3), strides=1, activation='relu', padding='same'))
    model.add(BatchNormalization())
    model.add(Conv2D(filters=128, kernel_size=(3, 3), strides=1, activation='relu', padding='same'))
    model.add(BatchNormalization())
    model.add(Conv2D(filters=128, kernel_size=(3, 3), strides=1, activation='relu', padding='same'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=2))

    model.add(Conv2D(filters=256, kernel_size=(3, 3), strides=1, activation='relu'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=2))

    model.add(Flatten())
    model.add(Dense(1024, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(43, activation='softmax'))

    model.load_weights('saved_models/weights.best.model.optimized.hdf5')

    return model

def get_traffic_sign_label(traffic_sign_index):
    traffic_signs_labels = np.load('traffic_sign_labels.npz')
    return traffic_signs_labels.f.traffic_sign_labels.item(0)[traffic_sign_index]

def predict_traffic_sign(img_path):
    # create instance of the CNN model
    model = create_model()
    # obtain predicted vector
    predicted_vector = model.predict(path_to_tensor(img_path))
    # return traffic sign that is predicted by the model
    return get_traffic_sign_label(np.argmax(predicted_vector))

@app.route('/predict', methods=['POST', 'GET'])
def predict():
    file = request.files['file']
    extension = os.path.splitext(file.filename)[1]
    fileName = os.path.splitext(file.filename)[0]
    f_name = fileName + extension
    image_path = os.path.join("static\\Images_Test", f_name)
    file.save(image_path)

    predicted_sign_label = predict_traffic_sign(image_path)

    prediction_result = 'Our classifier prediction for this image: ' + predicted_sign_label

    return jsonify({'prediction': predicted_sign_label,
                    'predictionResult': prediction_result,
                    'imagePath': image_path})

if __name__ == '__main__':
    # this will start a local server
    app.run(debug=True)