import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from tensorflow.keras.preprocessing import image
from flask import Flask, render_template, request
from werkzeug.utils import secure_filename

app = Flask(__name__)

def load_data(dataset_dir, img_size):
    images = []
    labels = []
    classes = sorted(os.listdir(dataset_dir))
    num_classes = len(classes)

    print("Classes found:", classes)

    for class_index, class_name in enumerate(classes):
        class_dir = os.path.join(dataset_dir, class_name)
        print("Processing class:", class_name, "at path:", class_dir)
        for image_name in os.listdir(class_dir):
            image_path = os.path.join(class_dir, image_name)
            print("Processing image:", image_path)
            try:
                image = cv2.imread(image_path)
                if image is None:
                    print(f"Error: Unable to read image '{image_path}'")
                    continue
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert to RGB
                image = cv2.resize(image, (img_size, img_size)) # Resize image
                images.append(image)
                labels.append(class_index)
            except Exception as e:
                print(f"Error processing image '{image_path}': {str(e)}")

    images = np.array(images)
    labels = np.array(labels)
    return images, labels, num_classes

def preprocess_data(images, labels, num_classes, test_size=0.4, random_state=42):
    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(images, labels, test_size=test_size, random_state=random_state)

    # Normalize pixel values to the range [0, 1]
    X_train = X_train.astype('float32') / 255.0
    X_test = X_test.astype('float32') / 255.0

    # Convert labels to one-hot encoding
    y_train = to_categorical(y_train, num_classes)
    y_test = to_categorical(y_test, num_classes)

    return X_train, X_test, y_train, y_test

# Function to preprocess uploaded image
def preprocess_uploaded_image(image_path, img_size):
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convert to RGB
    img = cv2.resize(img, (img_size, img_size)) # Resize image
    img = np.array(img)
    img = img.astype('float32') / 255.0
    return img

# Path to the dataset directory
dataset_dir = "D:/Downloads/Fruit_Disease_detection_With_GUI/Dataset/SenMangoFruitDDS_bgremoved"
# Image size for resizing
img_size = 128

# Load data and labels
images, labels, num_classes = load_data(dataset_dir, img_size)

# Preprocess the data
X_train, X_test, y_train, y_test = preprocess_data(images, labels, num_classes, test_size=0.4)

# Function to create and compile the CNN model
def create_model(input_shape, num_classes):
    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
        MaxPooling2D((2, 2)),
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        Conv2D(128, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        Flatten(),
        Dense(512, activation='relu'),
        Dropout(0.5),
        Dense(num_classes, activation='softmax')
    ])

    # Compile the model
    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    return model

# Input shape
input_shape = (img_size, img_size, 3) # Assuming RGB images
# Number of classes
num_classes = len(np.unique(labels))

# Create and compile the model
model = create_model(input_shape, num_classes)

# Display model summary
model.summary()

# Train the model
history = model.fit(X_train, y_train,
                    epochs=20,
                    batch_size=32,
                    validation_data=(X_test, y_test))

# Function to make predictions on uploaded image
def make_prediction(model, uploaded_img):
    prediction = model.predict(uploaded_img)
    predicted_class = np.argmax(prediction)
    return predicted_class

# Function to upload an image from local system
@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        file = request.files['file']
        if file:
            filename = secure_filename(file.filename)
            file_path = os.path.join("uploads", filename)
            file.save(file_path)

            # Preprocess uploaded image
            uploaded_img = preprocess_uploaded_image(file_path, img_size)

            # Reshape image for model prediction
            uploaded_img = np.expand_dims(uploaded_img, axis=0)

            # Make predictions
            predicted_class = make_prediction(model, uploaded_img)

            # Get class label
            classes = sorted(os.listdir(dataset_dir))
            predicted_label = classes[predicted_class]

            # Display prediction result
            return render_template('result.html', prediction=predicted_label, image=file_path)
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)

#http://127.0.0.1:5000/