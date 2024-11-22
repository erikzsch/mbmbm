# **Training Script for Bioinformatics OTU Data**

This repository provides a training script for processing and analyzing bioinformatics OTU data using custom pipelines built on top of `scikit-learn`, `PyTorch`, and `Hydra`. It includes configurable components for data preprocessing, transformation, feature engineering, model training, and evaluation.

---

## **Key Components**

### **1. Data Loading**
- **Purpose**: Load training and test datasets.
- **Key Method**: `init_data_provider`
- **Details**:
  - Provides training (`arr_train`, `arr_train_target`) and testing datasets (`arr_test`, `arr_test_target`).
  - Configuration-driven, supports flexible dataset formats.

---

### **2. Target Transforms**
- **Purpose**: Apply transformations specifically to the target variable.
- **Base Class**: `TransformBase`
- **Process**:
  - Instantiates target transformations from the `cfg.target_preproc` configuration.
  - Transforms the target data (`arr_train_target` and `arr_test_target`) while ensuring no rows or columns critical to testing are removed.

---

### **3. Input Transforms**
- **Purpose**: Preprocess and transform input features.
- **Base Class**: `TransformBase`
- **Process**:
  - Instantiates input transformations from `cfg.preproc`.
  - Applies transformations sequentially to `arr_train` and `arr_test`.
  - Handles removal of redundant rows and columns while maintaining compatibility between datasets.

---

### **4. Feature Engineering**
- **Purpose**: Enhance input data through feature selection or transformation.
- **Initialization**: `init_feature_engine`
- **Details**:
  - Configured using `cfg.feature_engine`.
  - Optionally loads a pre-trained feature engine from `cfg.feature_engine_dir`.
  - If none exists, fits the feature engine to the training data and saves its state.
  - Transforms training and test datasets after feature engineering.

---

### **5. Model Initialization**
- **Base Class**: `ModelBase`
- **Details**:
  - Configured via `cfg.model`.
  - Supports predefined models from `PREDEFINED_MODELS` or instantiates new models.
  - Handles classification or regression tasks based on `cfg.is_classif_model`.

---

### **6. Training and Prediction**
- **Purpose**: Train the model on the processed data and make predictions.
- **Process**:
  - Trains the model using `arr_train` and `arr_train_target`.
  - Saves the trained model to the output directory.
  - Predicts on the test dataset (`arr_test`).

---

### **7. Metrics and Evaluation**
- **Metrics Framework**: `MetricCollection`
- **Default Metrics**:
  - Mean Squared Error (MSE)
  - Mean Absolute Error (MAE)
  - R2 Score
- **Process**:
  - Configurable via `cfg.metrics`.
  - Evaluates predictions (`arr_test_predict`) against test targets (`arr_test_target`).
  - Logs and saves metric scores in CSV and JSON formats.

---

### **8. Logging and Plotting**
- **Logging**:
  - Logs training progress, execution times, and results to a file (`train.log`).
- **Execution Time Plotting**:
  - Visualizes time spent in different stages (e.g., data loading, transformation, model fitting) if plotting is enabled (`is_train_plotting=True`).

---

## **Outputs**
1. **Metrics**:
   - Evaluation scores (e.g., MSE, MAE, R2) saved as `metrics.csv` and `metrics.json`.
2. **Trained Model**:
   - Saved in the output directory for future use.
3. **Feature Engineering Results**:
   - Saved as state files to allow reuse or further analysis.
4. **Execution Time Plot**:
   - Optional visualization of pipeline timings.

---


