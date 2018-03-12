import _pickle as cPickle
import urllib
import cv2
import numpy as np
import io
import os
import tensorflow as tf
from keras import backend as K
from keras.layers import Conv2D, MaxPooling2D, GlobalAveragePooling2D
from keras.applications.inception_v3 import InceptionV3, preprocess_input
from keras.applications.xception import Xception, preprocess_input, decode_predictions
from keras.layers import Dropout, Flatten, Dense
from keras.models import Model
from keras.models import Sequential
from flask import Flask, jsonify, send_file, make_response, request, render_template
import urllib.request
import json
from sklearn.datasets import load_files
from keras.utils import np_utils
from glob import glob
from keras.applications.resnet50 import ResNet50
from keras.preprocessing import image
from keras.applications.resnet50 import preprocess_input, decode_predictions
#from PIL import Image
from keras.applications import *
from keras.callbacks import EarlyStopping, ModelCheckpoint
from extract_bottleneck_features import *
from tqdm import tqdm
from PIL import Image
from io import StringIO
import base64
import uuid
from werkzeug import secure_filename

app = Flask(__name__) # create a Flask app


@app.route('/')
def predictbreed():
   return render_template('DogBreedsPredictionApp.html')

def path_to_tensor(img_path):
    # loads RGB image as PIL.Image.Image type
    img = image.load_img(img_path, target_size=(224, 224))
    # convert PIL.Image.Image type to 3D tensor with shape (224, 224, 3)
    x = image.img_to_array(img)
    # convert 3D tensor to 4D tensor with shape (1, 224, 224, 3) and return 4D tensor
    return np.expand_dims(x, axis=0)


# load the user configs
with open('config/conf.json') as f:
	config = json.load(f)

# helper
def url_to_image(image_url, resize=224):
  """
  downloads an image from url, converts to numpy array,
  resizes, and returns it
  """
  with urllib.request.urlopen(image_url) as url:
      response = url.read()
  img = np.asarray(bytearray(response), dtype=np.uint8)
  img = cv2.imdecode(img, cv2.IMREAD_COLOR)
  img = cv2.resize(img, (resize, resize), interpolation=cv2.INTER_CUBIC)
  return img

def get_nnet(forTraining=True):
  #model = Xception(weights=weights)
  '''model = base_model.output
  # add a global spatial average pooling layer
  model = GlobalAveragePooling2D()(model)
  # add a fully-connected hiddden layer
  model = Dense(512, activation='relu')(model)
  # and a fully connected output/classification layer
  model = Dense(133, activation='softmax')(model)
  # create the full network so we can train on it
  final_model = Model(input=base_model.input, output=model)
  for layer in base_model.layers:
    layer.trainable = False'''
  if forTraining == True:
      ResNet50bottleneck_features = np.load('bottleneck_features/DogXceptionData.npz')
      train_ResNet50 = ResNet50bottleneck_features['train']
      valid_ResNet50 = ResNet50bottleneck_features['valid']
      #test_ResNet50 = ResNet50bottleneck_features['test']
  ResNet50_new_model = Sequential()
  ResNet50_new_model.add(GlobalAveragePooling2D(input_shape=(7, 7, 2048)))
  ResNet50_new_model.add(Dense(133, activation='softmax'))

  if forTraining == True:
      ResNet50_new_model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
      checkpointer = ModelCheckpoint(filepath='saved_models/weights.best.DogXception.hdf5',
                                 verbose=1, save_best_only=True)

      earlyStopping = EarlyStopping(patience=10)

      ResNet50_new_model.fit(train_ResNet50, train_targets,
                         validation_data=(valid_ResNet50, valid_targets),
                         epochs=100, batch_size=32, callbacks=[checkpointer, earlyStopping], verbose=2)
  if forTraining == False:
      ResNet50_new_model.load_weights('saved_models/weights.best.DogXception.hdf5')

  return ResNet50_new_model

def getBreedsLabels():
  return [item[20:-1] for item in sorted(glob("dogImages/train/*/"))]

def ResNet50_predict_breed(img_path):
  # extract bottleneck features
  ResNet50_new_model = get_nnet(forTraining=False)
  bottleneck_feature = extract_Xception(path_to_tensor(img_path))
  # obtain predicted vector
  predicted_vector = ResNet50_new_model.predict(bottleneck_feature)
  # return dog breed that is predicted by the model
  return getBreedsLabels()[np.argmax(predicted_vector)]

@app.route('/predict/<path:url>', methods=['POST'])
def predict(url):
  img = url_to_image(url) # image array

  # here to add some prep steps
  #img = image.load_img(url, target_size=(224, 224))
  #x = image.img_to_array(img)
  img = np.expand_dims(img, axis=0)
  #img = preprocess_input(img)
  model = get_nnet()
  preds = model.predict(img)
  img = Image.fromarray(np.squeeze(img, axis=(0,)), 'RGB')
  return jsonify({'predictions': str(decode_predictions(preds, top=1)[0][0][1])})

@app.route('/predict2/<path:imagePath>', methods=['POST'])
def predict2(imagePath):
    #image = cv2.imread(imagePath)
    #image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    #image = cv2.resize(image, (244, 244))
    predicted_dog_breed = ResNet50_predict_breed(imagePath)
    prediction_result = "Hello dog,\n Your predicted breed is \n" + predicted_dog_breed
    if predicted_dog_breed.lower() in imagePath.lower():
        prediction_result += ' (Correct)'
    else:
        prediction_result += ' (Wrong Perdiction)'
    return jsonify({'prediction': predicted_dog_breed, 'predictionResult': prediction_result})

@app.route('/predict3', methods=['POST', 'GET'])
def predict3():
    file = request.files['file']
    extension = os.path.splitext(file.filename)[1]
    fileName = os.path.splitext(file.filename)[0]
    #f_name = str(uuid.uuid4())+ "_" +  + extension
    f_name = fileName + extension
    imagePath = os.path.join("static\\Images_Test", f_name)
    file.save(imagePath)
    #image = cv2.imread(imagePath)
    #image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    #image = cv2.resize(image, (244, 244))
    predicted_dog_breed = ResNet50_predict_breed(imagePath)
    prediction_result = "Hello dog,\n Your predicted breed is \n" + predicted_dog_breed
    if predicted_dog_breed.lower() in imagePath.lower():
        prediction_result += ' (Correct)'
    else:
        prediction_result += ' (Wrong Perdiction)'
    #request.form['result'] = prediction_result
    return jsonify({'prediction': predicted_dog_breed,
                    'predictionResult': prediction_result,
                    'imagePath': imagePath})

@app.route('/resizeImg/<path:imagePath>', methods=['POST', 'GET'])
def resizeImage(imagePath):
    image = cv2.imread(imagePath)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, (224, 224))
    #pil_im = Image.fromarray(image)
    img_io = Image.open(imagePath)
    size = 224, 224
    img_io.thumbnail(size)
    img_io.save("Images_Test/Test_image.jpg", "JPEG")
    return send_file("Images_Test/Test_image.jpg", mimetype='image/jpeg')

def serve_pil_image(pil_img):
    img_io = StringIO()
    pil_img.save(img_io, 'JPEG', quality=70)
    img_io.seek(0)
    return send_file(img_io, mimetype='image/jpeg')

if __name__ == '__main__':
  print('initialize model...')
  model = get_nnet()
  print('load weights...')
  model.load_weights(model_weights_path)
  #print 'load label...'
  '''with open('label.pkl', 'rb') as handle:
    label = pickle.load(handle)'''

  app.run(debug=True) # this will start a local server