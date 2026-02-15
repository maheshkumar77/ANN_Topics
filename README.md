## 🩺 Diabetes Prediction Using Artificial Neural Network (ANN)

This project builds a **Deep Learning model (ANN)** to predict whether a patient has diabetes based on medical features. The model is trained on a diabetes dataset and evaluated using confusion matrix and classification metrics.

---

## 📌 Project Overview

Diabetes is a chronic disease that requires early detection for proper treatment. Machine Learning and Deep Learning models can help predict diabetes using patient health data.

In this project, we:

* Preprocess the dataset
* Train an Artificial Neural Network (ANN)
* Evaluate performance using Confusion Matrix and F1 Score
* Save the trained model for deployment (Flask/Web App)

---

## 📊 Dataset Information

The dataset contains patient medical attributes such as:

* Glucose level
* Blood pressure
* BMI
* Age
* Insulin
* Skin thickness
* Diabetes pedigree function
* Outcome (0 = Non-Diabetic, 1 = Diabetic)

---

## 🧠 Model Used: Artificial Neural Network (ANN)

Artificial Neural Networks mimic the human brain using layers of neurons.

### Architecture Example:

* Input Layer → Patient features
* Hidden Layers → Dense layers with ReLU activation
* Output Layer → Sigmoid activation for binary classification

---

## 📉 Model Evaluation

### ✅ Confusion Matrix

```
[[18279    13]
 [  555  1153]]
```

| Actual \ Predicted   | Non-Diabetic (0) | Diabetic (1) |
| -------------------- | ---------------- | ------------ |
| **Non-Diabetic (0)** | 18279            | 13           |
| **Diabetic (1)**     | 555              | 1153         |

---

### 📊 Classification Report

```
              precision    recall  f1-score   support

           0       0.97      1.00      0.98     18292
           1       0.99      0.68      0.80      1708

    accuracy                           0.97     20000
   macro avg       0.98      0.84      0.89     20000
```

### 🔍 Interpretation of Results

* **Accuracy:** 97% (Overall model correctness)
* **Precision (Diabetic):** 99% → Very few false positives
* **Recall (Diabetic):** 68% → Some diabetic patients are missed (false negatives)
* **F1-Score:** 0.80 → Balanced performance between precision and recall

⚠️ Note: High accuracy but lower recall means the dataset is imbalanced (more non-diabetic samples).

---

## 🛠️ Technologies Used

* Python
* TensorFlow / Keras
* NumPy, Pandas
* Scikit-learn
* Matplotlib / Seaborn
* Google Colab
* Flask (for deployment)

---

## 🚀 How to Run the Project

### 1️⃣ Install Dependencies

```bash
pip install numpy pandas scikit-learn tensorflow matplotlib
```

### 2️⃣ Train the Model

```bash
python train_ann.py
```

### 3️⃣ Save the Model

```python
model.save("diabetes_ann_model.h5")
```

### 4️⃣ Load Model in Flask

```python
from tensorflow.keras.models import load_model
model = load_model("diabetes_ann_model.h5")
```

---

## 📦 Files in the Project

```
📂 diabetes-ann-project
│── dataset.csv
│── train_ann.py
│── diabetes_ann_model.h5
│── app.py (Flask)
│── requirements.txt
│── README.md
```

---

## ⚠️ Future Improvements

* Handle class imbalance using SMOTE or class weights
* Improve recall using threshold tuning
* Use deeper neural networks or CNN/LSTM
* Deploy as a web application

---

## 👨‍💻 Author

**Mahesh Kumar Sahu**

Deep Learning & AI Enthusiast

---

## ⭐ If you like this project, give it a star on GitHub!
