# 🏦 Fake Currency Detection using Deep Learning 🚀  

This project implements a **deep learning model** for detecting **fake Indian currency notes** using a **GAN-style Discriminator**. The model has been trained on **real and fake note images** and achieves an impressive **94% test accuracy**.  

---

## 📌 **Project Features**  
✅ **CNN-Based Model** inspired by GAN Discriminators  
✅ **Trained on Indian Currency Dataset** with **Real & Fake Notes**  
✅ **High Accuracy (94%) on Test Data**  
✅ **Optimized for Deployment & Real-World Use**  

---

## 📂 **Dataset Structure**  
Link of Dataset: https://www.kaggle.com/datasets/jayaprakashpondy/indian-currency-dataset/data

The dataset is organized as follows:

```
/Dataset  
│── train
  |-real
  |-fake
│── test
  |-real
  |-fake 
```

---

## ⚙️ **Setup Instructions**  

### 1️⃣ **Clone the Repository**  
```bash
git clone https://github.com/yourusername/fake-currency-detection.git
cd fake-currency-detection
```

### 2️⃣ **Install Dependencies**  
```bash
pip install -r requirements.txt
```

### 3️⃣ **Run the Jupyter Notebook**  
```bash
jupyter notebook
```
Open `notebook.ipynb` and follow the training and testing steps.

---

## 🛠️ **How to Test the Model?**  
Once the model is trained, you can test it on new datasets:  

```python
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np

# Load the trained model
model = tf.keras.models.load_model("model/final_model.keras")

# Test on a new image
img_path = "test_samples/fake_500.jpg"
img = image.load_img(img_path, target_size=(128, 128))
img_array = image.img_to_array(img) / 255.0
img_array = np.expand_dims(img_array, axis=0)

# Predict
prediction = model.predict(img_array)
print("Prediction:", "Real" if prediction > 0.5 else "Fake")
```

---

## 📊 **Results (94% Test Accuracy)**  
✅ **Test Accuracy**: **94%**  
✅ **Precision (Fake Notes)**: **100%**  
✅ **Recall (Fake Notes)**: **90%**  
✅ **Confusion Matrix:**  
```
[[53  6]  # Fake Notes
 [ 0 48]] # Real Notes
```

---

## 🔗 **References**  
- [TensorFlow](https://www.tensorflow.org/)  
- [Keras Documentation](https://keras.io/)  

