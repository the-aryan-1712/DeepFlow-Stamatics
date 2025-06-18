
# Bulldozer Price Prediction (Major Deepflow)

This project focuses on predicting the sale price of construction equipment using structured data and machine learning. The model uses various features such as equipment year, machine hours, configuration types, and sale date to make accurate price estimates.

Some implementation choices were adapted to the environment — for example, avoiding local file downloads due to storage limits on macOS. The pipeline was designed to stay lean and interpretable, with minimal overengineering.

---

## Setup

* Developed and tested in Google Colab.
* Required packages:

  ```bash
  pip install -U gdown scikit-learn pandas matplotlib
  ```
* Data was downloaded from a hosted source using `gdown` and processed via Pandas.

---

## Workflow Summary

### 1. Data Loading

Compressed CSV files were read directly into memory. A brief inspection revealed the presence of missing values, categorical data, and mixed data types.

### 2. Data Cleaning & Feature Engineering

* Columns with excessive missing values were removed.
* Some categorical features were encoded, and missing entries were imputed with zeros or reasonable defaults.
* A key feature — the machine's operational age — was created by subtracting its manufacturing year from the sale year.

### 3. Model Development

* A `RandomForestRegressor` was used as the baseline model.
* Evaluation was based on RMSLE (Root Mean Squared Log Error).
* Hyperparameters were tuned using `RandomizedSearchCV`, leading to improved generalization and reduced error.

### 4. Feature Importance

Feature importance was extracted from the trained tree-based model. The most influential features in determining sale price included:

* `YearMade`
* `MachineHoursCurrentMeter`
* `ModelID`
* `ProductSize`

These features were cross-validated to ensure consistency in their impact.

### 5. Prediction Phase

A consistent structure was ensured between training and testing sets by:

* Filling missing columns with defaults
* Dropping unseen features
* Aligning column order before inference

This step resolved common real-world issues like schema mismatch during deployment.

---

## Results

The tuned model achieved the following:

* **Baseline RMSLE**: \~0.470
* **After tuning RMSLE**: \~0.321
* **Top-5 Features by Importance**:


  * MachineAge — Most important; machines older in age clearly lower in value.
  * ProductSize — Equipment size (Small, Medium, Large) affects price.
  * YearMade — Correlates with depreciation.
  * fiProductClassDesc — Describes product category/type.
  * ModelID — Proxy for model-specific effects not otherwise captured.


These results indicate that with appropriate preprocessing and careful handling of feature consistency, ensemble models can perform well even on noisy, mixed-type data.

---

## Final Notes

The notebook is kept simple and focused on functionality. While some steps (like deeper encoding or SHAP analysis) were skipped to stay within scope, the results are solid for a first iteration.

