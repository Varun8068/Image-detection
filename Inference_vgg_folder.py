import os
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array, load_img
import pandas as pd
import time

def load_and_preprocess_image(image_path, target_size=(224, 224)):
    img = load_img(image_path, target_size=target_size)
    img_array = img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array / 255.0  # Normalize pixel values to [0, 1]
    return img_array

def plot_confusion_matrix(conf_matrix, classes, save_path=None):
    plt.figure(figsize=(8, 6))
    plt.imshow(conf_matrix, interpolation='nearest', cmap='coolwarm')
    plt.title('Confusion Matrix', fontsize=16)
    plt.colorbar()

    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=20)
    plt.yticks(tick_marks, classes)

    plt.xlabel('Predicted Labels', fontsize=14)
    plt.ylabel('True Labels', fontsize=14)

    for i in range(len(classes)):
        for j in range(len(classes)):
            plt.text(j, i, str(conf_matrix[i, j]), ha='center', va='center', color='black', fontsize=12)

    if save_path:
        plt.savefig(save_path)
        print(f'Confusion matrix saved at: {save_path}')
    else:
        plt.show()

def perform_inference(model, test_folder, class_mapping, save_path=None, report_save_path=None):
    true_labels = []
    predicted_labels = []
    start_time = time.time()

    for class_folder in os.listdir(test_folder):
        class_path = os.path.join(test_folder, class_folder)
        if os.path.isdir(class_path):
            for image_file in os.listdir(class_path):
                image_path = os.path.join(class_path, image_file)

                # Load and preprocess the image
                img_array = load_and_preprocess_image(image_path)

                # Perform inference
                predictions = model.predict(img_array)
                predicted_class = np.argmax(predictions)

                # Map numeric labels to class names using the provided dictionary
                true_label = class_folder
                predicted_label = class_mapping[predicted_class]

                true_labels.append(true_label)
                predicted_labels.append(predicted_label)

    end_time = time.time()
    inference_time = end_time - start_time
    print(f'Inference Time: {inference_time:.2f} seconds')

    # Create confusion matrix
    conf_matrix = confusion_matrix(true_labels, predicted_labels)

    # Calculate accuracy
    accuracy = np.sum(np.diagonal(conf_matrix)) / np.sum(conf_matrix)
    print(conf_matrix)
    print(f'Accuracy: {accuracy * 100:.2f}%')

    # Print classification report
    class_labels = list(class_mapping.values())
    print('\nClassification Report:')
    report = classification_report(true_labels, predicted_labels, target_names=class_labels, output_dict=True)
    print(classification_report(true_labels, predicted_labels, target_names=class_labels))

    # Plot confusion matrix
    plot_confusion_matrix(conf_matrix, classes=class_labels, save_path=save_path)

    # Save classification report as an image
    if report_save_path:
        report_df = pd.DataFrame(report).transpose()
        plt.figure(figsize=(10, 6))
        heatmap = sns.heatmap(report_df, annot=True, cmap='viridis', cbar=False, fmt=".3f")
        plt.title('Classification Report')
        heatmap.get_figure().savefig(report_save_path)
        print(f'Classification report saved as an image at: {report_save_path}')

if __name__ == "__main__":
    # Load your pre-trained model
    model_path = r"C:\Users\paruc\PycharmProjects\Project1\brain_tumor_model_vgg16.h5"
    model = load_model(model_path)

    # Test folder containing subfolders for each class
    test_folder = r"C:\Users\paruc\Downloads\py files\brain tumour\Brain Tumor Data Set\Brain Tumor Data Set\val"

    # Class mapping dictionary
    class_mapping = {
        0: 'No Tumor',
        1: 'Tumor Present'
    }

    # Path to save the visualized confusion matrix
    save_path = 'brain_tumor_conf_mat.png'

    # Path to save the classification report heatmap
    report_save_path = 'brain_tumor_classification_report.png'

    # Perform inference and display confusion matrix, classification report, and accuracy
    perform_inference(model, test_folder, class_mapping, save_path, report_save_path)