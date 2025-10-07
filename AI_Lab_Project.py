import os
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from flask import Flask, request, render_template, jsonify

# Initialize Flask app
app = Flask(__name__)

# Paths to Your Datasets
base_dir = './datasets'
cats_and_dogs_dir = os.path.join(base_dir, 'cats_and_dogs')
flowers_dir = os.path.join(base_dir, 'flowers')

cats_and_dogs_train_dir = os.path.join(cats_and_dogs_dir, 'Train')
cats_and_dogs_val_dir = os.path.join(cats_and_dogs_dir, 'Valid')
cats_and_dogs_test_dir = os.path.join(cats_and_dogs_dir, 'Test')
cats_and_dogs_new_data_dir = os.path.join(cats_and_dogs_dir, 'NewData')

flowers_train_dir = os.path.join(flowers_dir, 'train')
flowers_test_dir = os.path.join(flowers_dir, 'test')
flowers_new_data_dir = os.path.join(flowers_dir, 'new_data')

# Create data generators
def create_generators(train_dir, val_dir=None, test_dir=None, target_size=(150, 150), batch_size=32):
    train_datagen = ImageDataGenerator(rescale=1./255, horizontal_flip=True, vertical_flip=True, rotation_range=20)
    val_datagen = ImageDataGenerator(rescale=1./255)
    test_datagen = ImageDataGenerator(rescale=1./255)

    train_generator = train_datagen.flow_from_directory(
        train_dir,
        target_size=target_size,
        batch_size=batch_size,
        class_mode='sparse'
    )

    val_generator = None
    if val_dir:
        val_generator = val_datagen.flow_from_directory(
            val_dir,
            target_size=target_size,
            batch_size=batch_size,
            class_mode='sparse'
        )

    test_generator = None
    if test_dir:
        test_generator = test_datagen.flow_from_directory(
            test_dir,
            target_size=target_size,
            batch_size=batch_size,
            class_mode='sparse'
        )

    return train_generator, val_generator, test_generator

cats_and_dogs_train_gen, cats_and_dogs_val_gen, cats_and_dogs_test_gen = create_generators(
    cats_and_dogs_train_dir, cats_and_dogs_val_dir, cats_and_dogs_test_dir
)
flowers_train_gen, _, flowers_test_gen = create_generators(flowers_train_dir, test_dir=flowers_test_dir)

# Debugging: Check the generators
print(f"cats_and_dogs_train_gen samples: {cats_and_dogs_train_gen.samples}")
print(f"cats_and_dogs_val_gen samples: {cats_and_dogs_val_gen.samples if cats_and_dogs_val_gen else 'None'}")
print(f"cats_and_dogs_test_gen samples: {cats_and_dogs_test_gen.samples if cats_and_dogs_test_gen else 'None'}")

print(f"flowers_train_gen samples: {flowers_train_gen.samples}")
print(f"flowers_test_gen samples: {flowers_test_gen.samples if flowers_test_gen else 'None'}")

# Function to create model
def create_model(num_classes):
    base_model = VGG16(weights='imagenet', include_top=False, input_shape=(150, 150, 3))
    for layer in base_model.layers:
        layer.trainable = False

    x = Flatten()(base_model.output)
    x = Dense(512, activation='relu')(x)
    x = Dense(num_classes, activation='softmax')(x)

    model = Model(inputs=base_model.input, outputs=x)
    model.compile(optimizer=Adam(learning_rate=0.001), loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model

# Create and train models
history = None
if not os.path.exists('cats_and_dogs_classifier.h5'):
    cats_and_dogs_model = create_model(len(cats_and_dogs_train_gen.class_indices))
    print("Training cats_and_dogs_model...")
    history = cats_and_dogs_model.fit(cats_and_dogs_train_gen, validation_data=cats_and_dogs_val_gen, epochs=10)
    cats_and_dogs_model.save('cats_and_dogs_classifier.h5')
else:
    cats_and_dogs_model = load_model('cats_and_dogs_classifier.h5')

if not os.path.exists('flowers_classifier.h5'):
    flowers_model = create_model(len(flowers_train_gen.class_indices))
    print("Training flowers_model...")
    history = flowers_model.fit(flowers_train_gen, validation_data=flowers_test_gen, epochs=10)
    flowers_model.save('flowers_classifier.h5')
else:
    flowers_model = load_model('flowers_classifier.h5')

# Flask routes
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    file = request.files['file']
    img_path = './uploads/' + file.filename
    file.save(img_path)

    dataset = request.form.get('dataset')
    if dataset == 'cats_and_dogs':
        model = cats_and_dogs_model
        class_indices = cats_and_dogs_train_gen.class_indices
    elif dataset == 'flowers':
        model = flowers_model
        class_indices = flowers_train_gen.class_indices
    else:
        return jsonify({'error': 'Invalid dataset selected'}), 400

    img_array = load_and_preprocess_image(img_path)
    predictions = model.predict(img_array)
    predicted_class_index = np.argmax(predictions, axis=1)[0]
    confidence = predictions[0][predicted_class_index]
    predicted_class_name = list(class_indices.keys())[predicted_class_index]

    return jsonify({
        'predicted_class': predicted_class_name,
        'confidence': float(confidence)
    })

def load_and_preprocess_image(img_path, target_size=(150, 150)):
    img = image.load_img(img_path, target_size=target_size)
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    img_array = img_array / 255.0  # Rescale to [0, 1]
    return img_array

# Plot training & validation accuracy and loss values
def plot_history(history, title_suffix=''):
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title(f'Model accuracy {title_suffix}')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend(['Train', 'Validation'], loc='upper left')

    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title(f'Model loss {title_suffix}')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend(['Train', 'Validation'], loc='upper left')
    plt.show()

if 'history' in locals() and history:
    plot_history(history)