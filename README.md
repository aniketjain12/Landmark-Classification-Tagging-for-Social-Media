
# Landmark Classification & Tagging for Social Media

## Overview

This project aims to classify landmarks from images and automatically infer their locations using Convolutional Neural Networks (CNNs). Photo sharing and storage services often utilize location data to enhance user experience by suggesting relevant tags or organizing photos. However, many images lack metadata with location information. This project addresses that challenge by detecting and classifying discernible landmarks in images.

The project involves building and comparing two different CNN architectures: one built from scratch and another using transfer learning. The best-performing model is then deployed as a web application, enabling users to upload images and automatically receive landmark classification and tagging.

## Table of Contents

- [Project Description](#project-description)
- [Dataset](#dataset)
- [Data Preprocessing](#data-preprocessing)
- [Model Architectures](#model-architectures)
  - [CNN from Scratch](#cnn-from-scratch)
  - [Transfer Learning](#transfer-learning)
- [Training & Evaluation](#training--evaluation)
- [Results](#results)
- [Deployment](#deployment)
- [How to Use](#how-to-use)
- [Run as a Standalone App](#run-as-a-standalone-app)
- [Dependencies](#dependencies)
- [Acknowledgments](#acknowledgments)

## Project Description

The objective of this project is to build a landmark classifier using deep learning techniques. The project workflow includes:
- **Data Collection:** Gathering a dataset of labeled landmark images.
- **Data Preprocessing:** Preparing the dataset by resizing images, normalizing pixel values, and augmenting data to improve model generalization.
- **Model Design:** Implementing two different CNN architectures: one from scratch and another using transfer learning with pre-trained models.
- **Training & Evaluation:** Training the models on the dataset and evaluating their performance.
- **Deployment:** Deploying the best-performing model as a web application.

## Dataset

The dataset used for this project consists of images of various landmarks, each labeled with the corresponding landmark name. The dataset was split into training, validation, and test sets to ensure robust model evaluation.

## Data Preprocessing

Data preprocessing is a crucial step to ensure that the input data is in a suitable format for the model. The following steps were performed:
1. **Resizing:** All images were resized to a fixed dimension (e.g., 224x224 pixels) to maintain consistency across the dataset.
2. **Normalization:** Pixel values were normalized to bring them to a common scale, typically between 0 and 1.
3. **Data Augmentation:** Techniques such as rotation, flipping, zooming, and random cropping were applied to the training data to increase model robustness and prevent overfitting.

## Model Architectures

### CNN from Scratch

A custom CNN was designed and implemented from scratch. The architecture includes:
- **Convolutional Layers:** To extract hierarchical features from images.
- **Pooling Layers:** To downsample the feature maps, reducing dimensionality and computation.
- **Fully Connected Layers:** To classify the images based on the extracted features.
- **Regularization:** Dropout was applied to prevent overfitting and improve generalization.

### Transfer Learning

Transfer learning was applied using a pre-trained model such as VGG16 or ResNet as the base. The top layers of the pre-trained model were fine-tuned, and additional fully connected layers were added to adapt the model for the specific task of landmark classification.

## Training & Evaluation

Both models were trained and evaluated using Jupyter notebooks:
- **Optimizer:** Adam or SGD optimizers were used to minimize the loss function.
- **Loss Function:** Categorical Cross-Entropy was used as the loss function for multi-class classification.
- **Evaluation Metrics:** The models were evaluated using accuracy, precision, recall, and F1-score on the validation and test sets.
- **Confusion Matrix:** A confusion matrix was generated to visualize the model's performance across different classes.

## Results

- **CNN from Scratch:** Achieved an accuracy of X% on the test set, demonstrating the ability to learn from scratch without relying on pre-trained knowledge.
- **Transfer Learning Model:** Achieved a higher accuracy of Y% on the test set, leveraging the power of pre-trained models to improve classification performance.

## Deployment

The best-performing model was deployed as a web application within a Jupyter notebook, allowing users to upload images and receive automatic landmark classification and tagging.

## How to Use

1. **Clone the Repository:**
   ```bash
   git clone https://github.com/aniketjain12/Landmark-Classification-Tagging-for-Social-Media.git
   ```
2. **Install Dependencies:**
   ```bash
   pip install -r requirements.txt
   ```
3. **Run the Jupyter Notebooks:**
   - Open the notebooks in Jupyter.
   - Run the cells sequentially to train the models, evaluate them, and deploy the web application.

4. **Access the Web Application:**
   The web application can be run directly from the notebook. Simply execute the deployment cells and follow the provided instructions to upload and classify images.

## Run as a Standalone App

You can run this notebook as a standalone app on your computer by following these steps:

1. **Download the Notebook:**
   Save this notebook in a directory on your machine.
   
2. **Download the Model Export:**
   Download the model export (e.g., `checkpoints/transfer_exported.pt`) into a subdirectory called `checkpoints` within the directory where you saved the notebook.

3. **Install Voila:**
   If you don't have Voila installed, you can install it with:
   ```bash
   pip install voila
   ```

4. **Run the App:**
   Use Voila to run the notebook as a standalone web app:
   ```bash
   voila app.ipynb --show_tracebacks=True
   ```

5. **Customize the App:**
   You can further customize your notebook to improve the app's interface and appearance, then rerun it with Voila.

## Dependencies

- Python 3.12
- Pytorch
- NumPy
- Pandas
- Matplotlib
- Voila

## Acknowledgments

This project was inspired by the need for automated image tagging in photo-sharing services. The CNN architectures were built using concepts learned from the Convolutional Neural Network course. Special thanks to the authors of the pre-trained models used for transfer learning.

