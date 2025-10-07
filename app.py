from flask import Flask, request, jsonify, render_template
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import os
import logging

app = Flask(__name__)

# Paths to the saved models
CATS_AND_DOGS_MODEL_PATH = './cats_and_dogs_classifier.h5'
FLOWERS_MODEL_PATH = './flowers_classifier.h5'

# Load the trained models
cats_and_dogs_model = load_model(CATS_AND_DOGS_MODEL_PATH)
flowers_model = load_model(FLOWERS_MODEL_PATH)

# Class names for each model
cats_and_dogs_class_names = ['Cat', 'Dog']
flowers_class_names = ['Daisy', 'Dandelion', 'Roses', 'Sunflowers', 'Tulips']

def load_and_preprocess_image(img_path, target_size=(150, 150)):
    img = image.load_img(img_path, target_size=target_size)
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array / 255.0
    return img_array

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        logging.error('No file part in the request')
        return jsonify({'error': 'No file part'})

    file = request.files['file']

    if file.filename == '':
        logging.error('No selected file')
        return jsonify({'error': 'No selected file'})

    if file:
        filepath = os.path.join('uploads', file.filename)
        file.save(filepath)
        
        dataset = request.form.get('dataset')
        if dataset == 'cats_and_dogs':
            model = cats_and_dogs_model
            class_names = cats_and_dogs_class_names
        elif dataset == 'flowers':
            model = flowers_model
            class_names = flowers_class_names
        else:
            logging.error('Invalid dataset selected')
            return jsonify({'error': 'Invalid dataset selected'})

        img_array = load_and_preprocess_image(filepath)
        prediction = model.predict(img_array)
        predicted_class = class_names[np.argmax(prediction[0])]
        confidence = float(np.max(prediction[0]))

        os.remove(filepath)
        return jsonify({'predicted_class': predicted_class, 'confidence': confidence})

    logging.error('File processing error')
    return jsonify({'error': 'File processing error'})

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    if not os.path.exists('uploads'):
        os.makedirs('uploads')
    app.run(debug=True)
