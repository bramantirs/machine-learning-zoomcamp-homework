1. Problem Definition & Scoping

Business problem: Fintech company want to predict the risk of loan default before approving a loan.

ML problem: Binary classification → default = 1, non-default = 0

Success metric:
    - ROC-AUC (as the main basis, if the data is imbalanced), 
    - Recall (minimize false negative), 
    - Precision (avoid incorrect rejection).

Deliverable: Model + insight which will help risk department

2. Data Loading & Initial Exploration 
  - check for possible dataset problem
  
3. Data Cleaning
  - check for possible data imbalance (features and target)
  - impute missing values with mean/median/mode
  - handle duplication and outlier
  - transform data type (datetime, category, integer, float, etc)
  - category encoding (labelencoder/onehot)
  - numerical scaling (standarscaler/minmaxscaler)
  
4. Extensive EDA
  - distribution of numerical and categorical features
  - feature colleration with other features
  - feature colleration with target
  - visualization
  - insight
  
5. Feature Engineering
  - create bins for credit score like 'is it low/medium/high?' which relevant with target
  - merge same features
  
6. Model Training & Evaluation
  - split data
  - model baseline
      - logistic regression
      - decision tree
      - random forest
      - xgboost
  - evaluate with roc-auc, recall, precission, f1-score, and confusion matrix
  - select the best model performance
  
7. Hypermeter Tuning & Validation
  - use gridsearchcv or randomsearchcv
  - k-fold (5-10 folds)
  - write down tuning result and execution time 
  - compare model performance (linear regression vs random forest vs xgboost)
  
8. Model Interpretation
  - feature importance
  - shap values
  - partial dependence plot
  - explain in business word (business translation)
  
9. Exporting Notebook to Script
  - separate moduls

10. Reproducibility & Dependency Management
  - create requirements.txt for user
  - save your all libraries version information
  - save random_state for function that used random_state

11. Containerization (Docker)
  - create dockerfile
  - build & run

12. Cloud Deployment 
  - streamlit cloud or hugging face spaces

13. Documentation
  - problem description
  - dataset summary
  - EDA findings
  - model performance
  - feature importance
  - business implication
  - Limitation & next step