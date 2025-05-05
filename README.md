[![Support Palestine](https://raw.githubusercontent.com/Ademking/Support-Palestine/main/Support-Palestine.svg)](https://www.map.org.uk)
# 📊 MNIST Digit Classification using Neural Network

![MNIST-Digit-Classification](ipc-thi-giac-may-tinh.jpg)

## Overview

> A simple Neural Network to classify handwritten digits from the MNIST dataset.

This project demonstrates how to build and train a basic neural network using TensorFlow/Keras to recognize handwritten digits (0–9) from the famous **MNIST dataset**. It includes preprocessing, model building, training, evaluation, visualization of results, and a predictive system for custom digit images.

---

## 📁 Dataset Used

The dataset is loaded directly from Keras:

```python
from tensorflow.keras.datasets import mnist
```

- ✅ 60,000 training images  
- ✅ 10,000 test images  
- ✅ Each image: `28x28` grayscale

Label range: **0 to 9** (digits)

---

## 🛠️ Technologies Used

- Python
- TensorFlow / Keras
- NumPy
- Matplotlib
- Seaborn
- OpenCV
- PIL

---

## 🧠 Model Architecture

A simple feedforward neural network:

```python
model = Sequential([
    Flatten(input_shape=(28, 28)),
    Dense(50, activation='relu'),
    Dense(50, activation='relu'),
    Dense(10, activation='softmax')
])
```

Compiled with:
- Optimizer: `Adam`
- Loss function: `sparse_categorical_crossentropy`
- Metric: `accuracy`

---

## 🏃‍♂️ How to Run

### 🔽 Clone the repo

```bash
git clone https://github.com/Abdallah-707/MNIST-Digit-Classification.git
cd MNIST-Digit-Classification
```

### ⚙️ Install dependencies

```bash
pip install numpy matplotlib seaborn tensorflow opencv-python pillow
```

### 📦 Download or use built-in dataset

No manual download required — dataset is automatically downloaded via Keras:

```python
(X_train, Y_train), (X_test, Y_test) = mnist.load_data()
```

You can also test with your own handwritten digit images.

---

## 📊 Training Summary

| Epochs | Train Acc | Test Acc |
|--------|-----------|----------|
| 10     | ~99%    | ~97%   |

Plots include:
- Sample image display
- Confusion matrix heatmap
- Prediction on custom input image

---

## 📈 Evaluation Metrics

After training, the model evaluates performance on the test set and displays:
- Accuracy
- Confusion Matrix
- Predictions vs True Labels

Example Confusion Matrix:
```python
conf_mat = confusion_matrix(Y_test, Y_pred_labels)
sns.heatmap(conf_mat, annot=True, fmt='d', cmap='Blues')
```

---

## 📷 Predictive System

Predict digits from custom images:

```python
input_image_path = input('Path of the image to be predicted: ')
input_image = cv2.imread(input_image_path)
grayscale = cv2.cvtColor(input_image, cv2.COLOR_RGB2GRAY)
resized = cv2.resize(grayscale, (28, 28)) / 255.0
reshaped = np.reshape(resized, [1, 28, 28])
prediction = model.predict(reshaped)
digit = np.argmax(prediction)

print('The Handwritten Digit is recognized as:', digit)
```

---

## 📋 Requirements File

Here’s a sample `requirements.txt`:

```
numpy
matplotlib
seaborn
tensorflow
opencv-python
pillow
```

Generate it using:
```bash
pip freeze > requirements.txt
```

---

## 🚀 Future Enhancements

- Add support for real-time webcam input
- Use CNN instead of dense layers for better accuracy
- Build a GUI using Streamlit or Tkinter
- Deploy as a web app using Flask or FastAPI

---

## 📄 License

MIT License – see [LICENSE](LICENSE)
