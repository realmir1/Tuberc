# Tuberculosis Chest X-Ray Classification

This project demonstrates the classification of chest X-ray images to distinguish between normal and tuberculosis-infected lungs using a Convolutional Neural Network (CNN) implemented in TensorFlow/Keras. The dataset used is the **TB Chest Radiography Database** from Kaggle. The code is designed to be run in a Kaggle environment.
<br>

<div align="center">
  <img src="https://miro.medium.com/v2/resize:fit:837/1*tI-TWV--K05xbXUgA4Qm1w.png" height="278" alt="python logo"  />
</div>

## Dataset
The dataset is sourced from the Tuberculosis Chest X-Ray Dataset available on Kaggle. It contains chest X-ray images categorized into two classes:

- **Normal**
- **Tuberculosis**

The data is preprocessed and split into training and validation subsets with a 90%-10% ratio using `ImageDataGenerator`.

## Model Architecture

The CNN model is implemented using TensorFlow/Keras. The architecture consists of the following layers:

1. **Input Layer**: Input shape of `(150, 150, 3)` for RGB images.
2. **Convolutional Layers**:
   - 3 convolutional layers with 32, 64, and 128 filters, each followed by ReLU activation.
3. **Pooling Layers**:
   - MaxPooling layers after each convolutional layer.
4. **Fully Connected Layers**:
   - A flattening layer followed by:
     - A dense layer with 128 units and ReLU activation.
     - An output layer with 1 unit and sigmoid activation for binary classification.

## Code Workflow

1. **Preprocessing**:
   - Images are rescaled to the range `[0, 1]` using `rescale=1.0/255.0`.
   - The data is split into training and validation subsets using `validation_split`.

2. **Model Compilation**:
   - **Optimizer**: Adam
   - **Loss Function**: Binary Crossentropy
   - **Metrics**: Accuracy

3. **Training**:
   - The model is trained for 10 epochs with a batch size of 32.

4. **Evaluation**:
   - Accuracy and loss curves are plotted.
   - Predictions are visualized on a subset of validation images.
   - A classification report is generated using `sklearn.metrics`.

## Visualizations

- **Accuracy and Loss Curves**:
  - Shows the training and validation accuracy and loss over epochs.
  
- **Prediction Samples**:
  - Displays predictions on 5 random images from the validation dataset, including the actual label and the model's predicted label.

## Results

The model generates a classification report with:

- **Precision**
- **Recall**
- **F1-score**
- **Support** for each class

## Usage

To run the project, follow these steps:

1. Clone the repository and upload it to Kaggle or a similar environment.
2. Download the dataset from Kaggle and place it in the appropriate directory (`/kaggle/input/tuberculosis-tb-chest-xray-dataset/TB_Chest_Radiography_Database`).
3. Execute the code in a Python environment with the required libraries installed.

## Dependencies

- Python
- TensorFlow
- NumPy
- Matplotlib
- scikit-learn

<div align="left">
  <img src="https://cdn.jsdelivr.net/gh/devicons/devicon/icons/python/python-original.svg" height="40" alt="python logo"  />
  <img width="12" />
  <img src="https://cdn.jsdelivr.net/gh/devicons/devicon/icons/numpy/numpy-original.svg" height="40" alt="numpy logo"  />
  <img width="12" />
  <img src="https://cdn.jsdelivr.net/gh/devicons/devicon/icons/tensorflow/tensorflow-original.svg" height="40" alt="tensorflow logo"  />
  <img src="https://upload.wikimedia.org/wikipedia/commons/thumb/0/01/Created_with_Matplotlib-logo.svg/2048px-Created_with_Matplotlib-logo.svg.png" height="40" alt="plotlib logo"  />
  <img width="12" />
  <img src="https://upload.wikimedia.org/wikipedia/commons/thumb/0/05/Scikit_learn_logo_small.svg/2560px-Scikit_learn_logo_small.svg.png" height="40" alt="plotlib logo"/>
  <img width="12" />
</div>

###




## Sample Output

### Accuracy and Loss Graphs
The model's performance across training epochs can be visualized with accuracy and loss curves.

### Predictions

| Actual Label | Predicted Label |
|---------------|-----------------|
| Normal        | Normal          |
| Tuberculosis  | Tuberculosis    |

## INFO

all rights reserved





###
