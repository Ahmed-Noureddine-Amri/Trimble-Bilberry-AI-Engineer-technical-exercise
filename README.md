# Trimble-Bilberry-AI-Engineer-technical-exercise

This repository contains the code for a deep learning model that classifies images into either "Road" or "Field" categories. The model is built using TensorFlow and Keras, employing the VGG16 architecture with transfer learning.


## Dependencies

* Python 3.6 or higher
* TensorFlow 2.4
* Matplotlib
* Numpy

## Repository Structure

The repository is organized with separate folders for the training dataset, test images, models, and summaries. The notebook contains the code to train, evaluate, and visualize the model's performance. The metrics.py file contains custom metrics, such as the F1 Score, which is not built into the used version of TensorFlow.
```

road-field-classification/
|
|-- dataset/                     # Training dataset folder, organized into subfolders for each class
|   |-- Field/
|   |-- Road/
|
|-- test_images/                 # Test images folder
|   |-- Field/
|   |-- Road/
|
|-- models/                      # Folder to save trained models
|   |-- road_field_classifier_vgg16.h5
|
|-- summaries/                   # Folder containing summary files related to the task
|   |-- Summary-A ConvNet for the 2020s.pdf
|   |-- Summary-Technical task.pdf
|
|-- Road_and_Field_Image_Classifier.ipynb   # Notebook to train and evaluate the model
|-- metrics.py                   # Custom metric F1 Score, used in the model
|-- README.md                    # This README file with instructions and information
```

## Usage

1. Ensure that the necessary libraries are installed.

2. Organize your dataset by placing the training images in the dataset folder and test images in the test_images folder. The dataset should be organized into subfolders for each class (e.g., 'Field' and 'Road').

3. Run the notebook to train the model on the dataset and evaluate it on the test images.

The notebook performs the following steps:

* Imports the required libraries and configures GPU memory growth.
* Loads and preprocesses the training and validation data.
* Creates a VGG16-based model and compiles it with binary crossentropy loss, and accuracy and F1 Score as metrics.
* Trains and evaluates the model
* Plots the loss, accuracy, and F1 score graphs.
* Loads, predicts and visualizes the results on the test images.
* Evaluates the model on the test dataset and prints the accuracy and F1 score.

## Notes

The model utilizes F1 Score as a performance metric, which is imported from the metrics.py file since it is not available in the TensorFlow version being used. To use the built-in F1 Score metric, the TensorFlow Addons package must be installed.
```
pip install tensorflow-addons
```

## License

[MIT](https://choosealicense.com/licenses/mit/)
