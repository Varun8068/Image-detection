from PIL import Image
import numpy as np
from keras.models import load_model

# Load the pre-trained model
model = load_model(r"C:\Users\paruc\PycharmProjects\Project1\brain_tumor_model_vgg19.h5")

# Define the class labels
classes = {
    0: 'No Tumor',
    1: 'Tumor Present'
}

def classify_tumor(image_path):
    # Load and preprocess the image
    image = Image.open(image_path)
    image = image.resize((224, 224))  # Adjust the size as per your model's input
    image = np.array(image) / 255.0  # Normalize the pixel values
    image = np.expand_dims(image, axis=0)  # Add a batch dimension

    # Make the prediction
    prediction = model.predict(image)
    print(f"Prediction: {prediction}")

    # Get the class label
    if prediction[0][0] > 0.5:
        label = classes[1]
    else:
        label = classes[0]

    print(f"Classification: {label}")

# Example usage
image_path = r"C:\Users\paruc\Downloads\Programs Ai\no\no915.jpg"
classify_tumor(image_path)