# Student Exam Score Prediction

Machine Learning project for predicting a student's **Exam Score** from academic, personal, and school-related features.

The project compares four regression algorithms and integrates the final trained model into a **Streamlit** application for prediction.

## 1. Project Objective

The objective is to build a reproducible machine-learning workflow that:

* explores and cleans the dataset;
* prepares numerical, nominal, and ordinal features;
* trains several regression models;
* tunes their hyperparameters using cross-validation;
* evaluates predictions on an independent test set;
* packages preprocessing and the model into a single pipeline;
* provides a user interface with Streamlit.

## 2. Dataset

The dataset contains student information such as:

* Hours Studied
* Attendance
* Parental Involvement
* Access to Resources
* Extracurricular Activities
* Sleep Hours
* Previous Scores
* Motivation Level
* Internet Access
* Tutoring Sessions
* Family Income
* Teacher Quality
* School Type
* Peer Influence
* Physical Activity
* Learning Disabilities
* Parental Education Level
* Distance from Home
* Gender

### Target

`Exam_Score`

The target represents the student's exam score and is treated as a continuous numerical variable, making this a **regression problem**.

## 3. Data Preparation

The data preparation process includes:

1. Inspecting the dataset structure and data types.
2. Detecting missing values.
3. Handling missing categorical values using the mode.
4. Detecting and removing duplicated observations.
5. Investigating outliers in `Exam_Score`.
6. Splitting the data into training and test sets.

The dataset is split using:

```python
train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=10
)
```

## 4. Feature Transformation

The preprocessing is performed inside a Scikit-learn `Pipeline` using a `ColumnTransformer`.

### Numerical features

Standardized with `StandardScaler`:

* `Hours_Studied`
* `Attendance`
* `Sleep_Hours`
* `Previous_Scores`
* `Tutoring_Sessions`
* `Physical_Activity`

### Nominal features

Encoded using `OneHotEncoder`:

* `Gender`
* `School_Type`
* `Extracurricular_Activities`
* `Internet_Access`
* `Learning_Disabilities`
* `Peer_Influence`

### Ordinal features

Encoded using `OrdinalEncoder` because their categories have a natural order:

* `Parental_Involvement`
* `Access_to_Resources`
* `Motivation_Level`
* `Family_Income`
* `Teacher_Quality`
* `Parental_Education_Level`
* `Distance_from_Home`

The preprocessing is included in the model pipeline so that the same transformations are automatically applied during training, evaluation, and prediction.

## 5. Machine Learning Models

Four regression algorithms are studied:

### Linear Regression

A linear model used as a baseline for understanding the relationship between the features and exam score.

### Random Forest Regressor

An ensemble of decision trees capable of modelling nonlinear relationships and interactions between features.

### XGBoost Regressor

A gradient-boosting algorithm based on sequentially optimized decision trees.

### Support Vector Regression (SVR)

The regression version of Support Vector Machines, useful for modelling nonlinear relationships through kernels.

## 6. Hyperparameter Tuning

`GridSearchCV` is used to search for suitable hyperparameter configurations.

Three-fold cross-validation is used:

```python
KFold(
    n_splits=3,
    shuffle=True,
    random_state=10
)
```

The optimization metric is:

```python
scoring="r2"
```

The test set is kept separate from the cross-validation process and is used only for final evaluation.

## 7. Evaluation

The models are evaluated using:

### MAE — Mean Absolute Error

Measures the average absolute difference between predicted and actual scores.

### RMSE — Root Mean Squared Error

Measures prediction error while giving more weight to larger errors.

### R² — Coefficient of Determination

Measures how much of the variance in the target is explained by the model.

Residual analysis and predicted-vs-actual plots are also used to investigate model behaviour.

## 8. Project Structure

```text
edu_predict/
│
├── data/
│   └── dataset.csv
│
├── notebooks/
│   ├── 01_exploration.ipynb
│   ├── 02_profiling.ipynb
│   └── ...
│
├── src/
│   └── edu_predict/
│       ├── app.py
│       └── student_exam_model.joblib
│
├── pyproject.toml
├── uv.lock
├── .python-version
└── README.md
```

## 9. Model Persistence

The complete preprocessing and prediction pipeline is saved with `joblib`.

The saved object contains:

* the preprocessing pipeline;
* the trained model;
* the expected feature columns.

This allows the Streamlit application to use exactly the same preprocessing logic as the training process.

## 10. Streamlit Application

The application provides a form where the user enters student information.

The application then:

1. collects the input values;
2. constructs a Pandas DataFrame;
3. preserves the expected feature order;
4. sends the data to the saved pipeline;
5. displays the predicted exam score.

Run the application from the project root:

```bash
uv run streamlit run src/edu_predict/app.py
```

## 11. Reproducibility

The project uses `uv` for Python environment and dependency management.

The intended Python environment is:

```text
Python 3.13
scikit-learn 1.6.1
joblib 1.6.0
```

The same Scikit-learn version used to train the serialized model should be used when loading it in the Streamlit application.

## 12. Learning Approach

The project follows an academic machine-learning workflow:

```text
Problem
   ↓
Data Understanding
   ↓
Data Quality Analysis
   ↓
Data Cleaning
   ↓
Feature Definition
   ↓
Train/Test Split
   ↓
Preprocessing
   ↓
Baseline Models
   ↓
Cross-Validation
   ↓
Hyperparameter Tuning
   ↓
Final Evaluation
   ↓
Model Persistence
   ↓
Streamlit Deployment
```

The main objective is not only to obtain predictions, but to understand **why each preprocessing step, model, hyperparameter, and evaluation metric is used**.
