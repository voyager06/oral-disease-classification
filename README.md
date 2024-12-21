Oral Disease Detection Using VGG16
Overview
This project aims to classify oral diseases using deep learning techniques, achieving a notable accuracy improvement through the use of pre-trained convolutional neural network architectures. The model is trained on labeled images of oral conditions and demonstrates its effectiveness in differentiating between disease categories.

Dataset
The dataset consists of two primary subsets:

Training Set: Located in /content/images/TRAIN.
Test Set: Located in /content/images/TEST.
Each subset contains labeled directories representing different oral disease categories.

Model Architectures and Performance
Three different pre-trained architectures were used during experimentation:

ImageNet-based Model: Achieved an accuracy of 65%.
ResNet Architecture: Achieved an accuracy of 77%.
VGG16 with Transfer Learning: Achieved a final test accuracy of 95%, making it the most successful model in this study.
Implementation Details
Data Preprocessing:

Images resized to 224x224.
Data augmentation techniques, including random flipping and rotation, were applied.
Model Architecture:

The final model is based on VGG16 with pre-trained ImageNet weights.
The base model was frozen to preserve learned features, and a custom fully connected head was added:
Flatten layer
Dense layer with 512 neurons and ReLU activation
Dropout layer (50%)
Output layer with 4 neurons (softmax activation for multi-class classification).
Training:

Optimizer: Adam with a learning rate of 0.0001.
Loss Function: Sparse categorical cross-entropy.
Batch size: 32.
Number of epochs: 10.
Evaluation:

Confusion matrices were generated for both training and testing datasets to assess model performance.
Heatmaps were plotted to visualize the matrices.
Usage
Training
To train the model on the provided dataset:

Extract the dataset to the specified directories.
Run the script to train the model.
The model is saved as oral_disease_model.h5.
Prediction
To predict the class of a new image:

Load the model using TensorFlow.
Preprocess the input image:
Resize to 224x224.
Normalize pixel values.
Pass the image to the model for predictions.
Example
python
Copy code
# Define class labels
class_labels = ['Caries', 'Gingivitis']

# Predict
predictions = model.predict(image_batch)
predicted_index = np.argmax(predictions[0])
predicted_label = class_labels[predicted_index]
confidence = predictions[0][predicted_index]

print(f"Predicted Class: {predicted_label}, Confidence: {confidence:.2f}")
Results
The VGG16 model demonstrates state-of-the-art performance with a test accuracy of 95%.
The classification system is capable of identifying oral diseases with high confidence and reliability.
Dependencies
Python 3.x
TensorFlow/Keras
OpenCV
Matplotlib
Seaborn
Scikit-learn
License
This project is licensed under the MIT License. See the full license text below:

sql
Copy code
MIT License

Copyright (c) 2024 Vedant Swami.

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
