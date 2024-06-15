from sklearn.metrics import classification_report, confusion_matrix
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from io import StringIO
from ultralytics import YOLO
import os

# Load a pretrained YOLOv8n-cls model
model = YOLO(r"C:\Users\paruc\runs\classify\train\weights\best.pt")

# Define the path to the image folder
image_folder = r"C:\Users\paruc\Downloads\py files\brain tumour\Brain Tumor Data Set\Brain Tumor Data Set\val"

# Get a list of all subfolders in the main image folder
subfolders = [f.path for f in os.scandir(image_folder) if f.is_dir()]

# Create an empty list to store the results
all_results = []

# Run inference on each image and store the results
for subfolder in subfolders:
    actual_class = os.path.basename(subfolder)  # Get the actual class from the subfolder name
    image_files = [os.path.join(subfolder, f) for f in os.listdir(subfolder) if
                   os.path.isfile(os.path.join(subfolder, f))]

    for image_file in image_files:
        results = model(image_file)  # Get results for the current image
        for r in results:
            class_index = r.probs.top1  # Get the index of the top class
            if class_index == 0:  # If "tumor" class
                class_label = 0
                class_name = "Tumor Present"
            else:  # If "no_tumor" class
                class_label = 1
                class_name = "No Tumor"
            all_results.append({'image_name': os.path.basename(image_file), 'actual_class': actual_class,
                                'predicted_class': class_name})

# Convert the results to a DataFrame
df = pd.DataFrame(all_results)

# Get actual and predicted classes
actual_classes = df['actual_class']
predicted_classes = df['predicted_class']

# Create a classification report
class_report = classification_report(actual_classes, predicted_classes, target_names=['Tumor Present', 'No Tumor'])
print("Classification Report:\n", class_report)

# Create a confusion matrix
conf_matrix = confusion_matrix(actual_classes, predicted_classes, labels=['Tumor Present', 'No Tumor'])

# Plot both Classification Report Metrics and Confusion Matrix
plt.figure(figsize=(15, 6))

# Plot Confusion Matrix
plt.subplot(1, 2, 1)
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=['Tumor Present', 'No Tumor'],
            yticklabels=['Tumor Present', 'No Tumor'])
plt.xlabel('Predicted Class')
plt.ylabel('Actual Class')
plt.title('Confusion Matrix')

# Plot Classification Report
plt.subplot(1, 2, 2)
plt.text(0.1, 0.5, class_report, {'fontsize': 12}, va="center", ha="left")
plt.axis('off')
plt.title('Classification Report')

plt.tight_layout()
plt.show()