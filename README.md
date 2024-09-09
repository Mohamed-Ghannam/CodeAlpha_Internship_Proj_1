# Handwritten Digits Recognition

This project involves building a deep learning model to recognize handwritten digits using the MNIST dataset. The dataset contains 70,000 grayscale images of handwritten digits (0-9), each of size 28x28 pixels. The goal of the project is to train a Convolutional Neural Network (CNN) to classify these digits with high accuracy.

## Project Structure

- **1. Importing Libraries:**  
  The project begins by importing the necessary libraries such as TensorFlow, Keras, NumPy, Matplotlib, and Seaborn for model building, data processing, and visualization.

- **2. Loading the Dataset:**  
  The MNIST dataset is loaded from Keras datasets. The dataset is divided into training and testing sets: 60,000 images for training and 10,000 images for testing.

- **3. Data Exploration:**  
  The dataset is explored by displaying one instance of each unique label (0-9) and extracting random samples from the training set. This helps to understand the distribution and nature of the data.

- **4. Data Preprocessing:**  
  The data is preprocessed before modeling by normalizing the pixel values to a range of [0, 1]. Additionally, an extra dimension is added to the input data to make it compatible with the CNN's expected input shape.

- **5. Building the Model:**  
  A Convolutional Neural Network (CNN) is constructed with multiple layers, including convolutional layers, pooling layers, and dense layers. The architecture is designed to capture spatial hierarchies in the image data and optimize classification performance.

- **6. Model Evaluation:**  
  The model's performance is evaluated using various metrics:
  - **Training and Validation Accuracy/Loss:**  
    The training process is monitored by plotting the training and validation accuracies and losses over epochs.
  - **Model Summary:**  
    The summary of the model architecture, including the number of parameters, is displayed.
  - **Test Accuracy and Loss:**  
    The model is evaluated on the test set to calculate its accuracy and loss on unseen data.
  - **Confusion Matrix:**  
    A confusion matrix is plotted to visualize the performance of the model across different classes.

- **7. Predicting Random Images:**  
  A custom function is implemented to predict a random image from the test set. The function displays the image along with its true and predicted labels, including the model's confidence percentage for the prediction.

## Results

- **Test Accuracy:** The final test accuracy achieved by the model on the MNIST dataset.
- **Confusion Matrix:** A visual representation of the model's performance across all digit classes.
- **Sample Predictions:** Random sample predictions showing true labels, predicted labels, and the model's confidence levels.

## Conclusion

This project successfully builds and trains a CNN model to classify handwritten digits with high accuracy. The model evaluation metrics and confusion matrix indicate strong performance, and the prediction function provides a way to visualize the model's decision-making process.

## Dependencies

- Python 3.x
- TensorFlow
- Keras
- NumPy
- Matplotlib
- Seaborn
