import matplotlib.pyplot as plt
from PIL import Image
import os
from ultralytics import YOLO

# Function to load your YOLOv8 model
def load_model():
    # Load your YOLOv8 model here
    model = YOLO(r"C:\Users\paruc\runs\classify\train\weights\best.pt")  # Replace with your model's path
    return model

# Function to perform inference on an image
def perform_inference_on_image(model, image_path):
    if not os.path.exists(image_path):
        print(f"Error: The image file '{image_path}' does not exist.")
        return

    # Perform inference
    results = model(image_path)
    
    # Print the results to understand its structure
    print(results)
    
    # Check if results is a list and inspect the first item
    if isinstance(results, list) and len(results) > 0:
        result = results[0]
        print(result)
        
        # Check if the result has probabilities
        if hasattr(result, 'probs'):
            # Get the probabilities and class names
            probs = result.probs
            names = result.names
            
            # Print the probabilities and class names
            print(f"Top-1 Class: {names[probs.top1]}, Probability: {probs.top1conf.item()}")
            
            for idx, (class_idx, class_conf) in enumerate(zip(probs.top5, probs.top5conf)):
                print(f"Top-{idx+1} Class: {names[class_idx]}, Probability: {class_conf.item()}")
            
            # Display the image with the classification result
            img = Image.open(image_path)
            plt.imshow(img)
            plt.title(f"Top-1 Class: {names[probs.top1]}, Probability: {probs.top1conf.item():.2f}")
            plt.axis('off')
            plt.show()
        else:
            print("The result does not have probabilities.")
    else:
        print("The results object is not a list or it is empty.")

# Load the model
model = load_model()

# Provide the path to your image (now a .jpg file)
image_path = r"C:\Users\paruc\Downloads\py files\brain tumour\Tumor\Cancer (481).jpg"

# Call the function to perform inference on the image
perform_inference_on_image(model, image_path)
