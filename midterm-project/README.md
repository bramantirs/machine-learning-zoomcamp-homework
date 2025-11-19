# Loan Approval Prediction

## Overview

This project builds a complete machine learning pipeline to predict whether a loan application will be approved. The dataset contains realistic financial and demographic information of borrowers from the US and Canada.

### Dataset:
[Realistic Loan Approval Dataset (US and Canada)](https://www.kaggle.com/datasets/parthpatel2130/realistic-loan-approval-dataset-us-and-canada)

The goal of this project is to build a complete and reproducible machine learning workflow that can help improve the loan approval process. The workflow includes exploration, preprocessing, modeling, environment control, and containerization so the entire system can run the same way anywhere.

## Problem Description

Financial institutions need to decide which customers are likely to repay their loans. This decision affects revenue, risk, and overall portfolio health. When the review process is done only by hand, the results can take a long time and often depend on the reviewer. This creates delays for customers and uncertainty for the business.

A prediction system helps the business by offering a consistent way to evaluate risk and speed up the approval pipeline. By studying historical borrower data and learning the patterns behind good and bad loans, the model can support staff with fast and data driven suggestions.

This project explores the data, builds several models, and prepares the whole workflow so it can be repeated and used in a real operational setting. The focus is on understanding the patterns, improving decision quality, and building a pipeline that is ready for deployment when needed.

### Dataset Description

Key features included in the dataset:

- Identifier:

Customer ID (unique identifier for each application)

- Demographics:

Age, Occupation Status, Years Employed

- Financial Profile:

Annual Income, Credit Score, Credit History Length -Savings/Assets, Current Debt

- Credit Behaviour:

Defaults on File, Delinquencies, Derogatory Marks

- Loan Request:

Product Type, Loan Intent, Loan Amount, Interest Rate

- Calculated Ratios:

Debt-to-Income, Loan-to-Income, Payment-to-Income

• Loan status (target)

The dataset provides enough signal to build a realistic credit risk model.

## Project Workflow
### 1. Exploratory Data Analysis (EDA)

Performed inside the Jupyter notebook:

Data inspection and summary statistics

Missing values and data cleaning

Correlation and feature relationships

Categorical and numerical feature distributions

Target behavior analysis

Notebook:
```notebook/data_analysis-and-modelling.ipynb```

### 2. Reproducibility

To ensure consistent results across machines and future reruns:

Fixed random seeds

Tracked all preprocessing steps

Saved encoding artifacts (DictVectorizer)

Saved trained models in the ```models/``` directory

### 3. Model Training

Multiple machine learning models were trained and compared using consistent preprocessing and evaluation metrics.

Models used:

**Logistic Regression**

Scores: 

Accuracy: 0.8661

F1 Score: 0.8792

AUC: 0.9448

**Random Forest** 

Parameters tested included various max_depth and n_estimators.
Best configuration:

```max_depth = 20```

```n_estimators = 100```

Scores:

Accuracy: 0.9139

F1 Score: 0.9219

AUC: 0.9764

**CatBoost (Best Model)**

Well suited for tabular data with categorical variables.
Scores:

Accuracy: 0.9317

F1 Score: 0.9384

AUC: 0.9857


Training pipeline:
```scripts/train.py```

All models and encoders are saved to:
```
models/
├── dv.pkl
├── model_rf.pkl
├── model_lr.pkl
├── model_cb.cbm
```
### 4. Dependency and Environment Management

The project uses uv for dependency management and reproducibility.

Files:

pyproject.toml

uv.lock

Install dependencies:
```
uv sync
```
This recreates the exact environment used during development.

### 5. Packaging the Model for Deployment (Local Deployment)

A simple prediction script (inside your project) loads the model and encoder to generate predictions.
This is sufficient for teams who want to integrate the model into internal systems or batch pipelines.

### 6. Containerization with Docker

A Dockerfile is included for packaging the full environment and inference system.

Steps:

Build the image
```
docker build -t loan-approval-app .
```

Run the container
```
docker run -p 9696:9696 loan-approval-app
```

The container loads the trained model and exposes it through the prediction script.

This ensures consistent execution across any machine that supports Docker.
Project Structure
```
midterm-project/
│
├── data/
│   └── Loan_approval_data_2025.csv
│
├── models/
│   ├── dv.pkl
│   ├── model_rf.pkl
│   ├── model_lr.pkl
│   ├── model_cb.cbm
│
├── notebook/
│   └── data_analysis-and-modelling.ipynb
│
├── scripts/
│   └── train.py
│
├── Dockerfile
├── pyproject.toml
├── uv.lock
└── README.md
```
## How to Run the Project
Step 1: Install dependencies
```
uv sync
```
Step 2: Train the model
```
python scripts/train.py
```
Step 3: Build Docker container
```
docker build -t loan-approval-app .
```
Step 4: Run container
```
docker run -p 9696:9696 loan-approval-app
```

