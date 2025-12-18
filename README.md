# Credit Card Fraud Detection using Deep Neural Networks (DNN)

**Problem Type:** Binary Classification  
**Key Challenge:** Extreme class imbalance (fraud ≈ 0.17%)

This project demonstrates building a Deep Neural Network (DNN) to detect credit card fraud using the [Kaggle Credit Card Fraud Detection dataset](https://www.kaggle.com/mlg-ulb/creditcardfraud). The dataset contains anonymized features (V1–V28 via PCA), along with `Time`, `Amount`, and the target variable `Class` (0: Normal, 1: Fraud).

---

## **Project Overview**

### 1. Libraries Used

- **Data Handling & Visualization:** `numpy`, `pandas`, `matplotlib`, `seaborn`
- **Preprocessing & Metrics:** `sklearn`
- **Deep Learning:** `tensorflow`, `keras`
- **Explainability:** `shap`

### 2. Dataset

- Total samples: **284,807**
- Features: **30** (`V1–V28`, `Time`, `Amount`)
- Target: **Class** (fraudulent or non-fraudulent)
- Class distribution:
  - Non-fraud: 99.83%
  - Fraud: 0.17%

---

### 3. Preprocessing

- Scaled `Amount` and `Time` using `StandardScaler`
- Split dataset: 80% training, 20% testing (`stratify=y`)
- Handled class imbalance using **class weights**:
  ```python
  Class weights: {0: 0.50, 1: 289.14}
  ```

````

---

### 4. Model Architecture

* **Input layer:** 30 features
* **Hidden layers:**

  * Dense(32, ReLU) + Dropout(0.3)
  * Dense(16, ReLU) + Dropout(0.2)
* **Output layer:** Dense(1, Sigmoid)
* **Loss function:** Binary Crossentropy
* **Optimizer:** Adam
* **Metric:** AUC

**Total parameters:** 1,537

---

### 5. Training

* **Epochs:** 50
* **Batch size:** 2048
* **Early stopping** on validation loss (patience=5)
* **Validation split:** 20% of training data

**Training Results:**

* Best **ROC-AUC**: ~0.9765
* High **Recall** on fraud cases: 0.89
* Precision-Recall AUC: ~0.608

---

### 6. Evaluation

* **Confusion Matrix** and **Classification Report**
* **ROC-AUC** and **Precision-Recall curves**
* **SHAP** for model interpretability (feature importance)

---

### 7. Saving the Model

* Model: `credit_card_detection.h5`
* Scaler: `scaler.pkl`

---

### 8. Key Observations

* Extreme class imbalance requires **class weighting** or oversampling techniques.
* Model achieves high recall but low precision on fraud cases.
* SHAP highlights which features contribute most to fraud detection.

---

### 9. Future Improvements

* Use **SMOTE** or **ADASYN** for synthetic oversampling
* Explore **ensemble models** (XGBoost, LightGBM)
* Fine-tune **hyperparameters** for better precision

---

### 10. How to Run

```bash
# Clone repository
git clone <repo_url>
cd credit-card-fraud-detection

# Install dependencies
pip install -r requirements.txt

# Run training
python train_model.py

# Evaluate model
python evaluate_model.py
```

---

### 11. References

* Kaggle Dataset: [Credit Card Fraud Detection](https://www.kaggle.com/mlg-ulb/creditcardfraud)
* SHAP Documentation: [https://shap.readthedocs.io](https://shap.readthedocs.io)
````
