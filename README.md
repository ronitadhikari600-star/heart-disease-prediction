# Heart Disease Prediction and Patient Risk Analysis

A machine learning project that predicts whether a patient is likely to have **Heart Disease (Yes/No)**, gives a **patient risk score**, and assigns a **risk category (Low, Medium, High)**.

## Dataset
`heart_disease_risk.csv` has 70,000 patient records with 18 features. The target column `Heart_Risk` is 0 for no risk and 1 for risk. After removing duplicates, 63,755 records are used. The features are 8 symptoms, 8 risk factors, Gender and Age.

## Project Steps
1. Load and understand the dataset
2. Preprocess the data (missing values, duplicates, outliers)
3. Exploratory data analysis (10 graphs)
4. Select the features with more than 50% correlation with `Heart_Risk` (Age and 8 symptoms)
5. Train/test split (80/20, stratified) and scaling
6. Build four models: Logistic Regression, Linear Regression, KNN, K-Means
7. 5-fold cross-validation and hyperparameter tuning with GridSearchCV
8. Enter patient data to get the heart disease prediction, risk score and risk category
9. Save the models with joblib

## Models
| Model | Purpose |
|---|---|
| Logistic Regression | Predict heart disease (Yes/No) |
| Linear Regression | Predict the patient risk score |
| KNN | Find patients with similar medical conditions |
| K-Means | Group patients into Low, Medium and High risk |

## Results
The selected model for the Yes/No prediction is a tuned Logistic Regression (`C=1`).

| Metric (test set) | Score |
|---|---|
| Accuracy | 0.9792 |
| Precision | 0.98 |
| Recall | 0.98 |
| F1-score | 0.98 |
| Cross-validation accuracy | 0.9769 |

KNN reaches about 0.978 accuracy. The Linear Regression risk score has an R2 of 0.83.

## How to Run
1. Install the libraries:
   ```
   pip install numpy pandas matplotlib seaborn scikit-learn joblib jupyter
   ```
2. Keep `heart_disease_risk.csv` and the notebook in the same folder.
3. Open `heart_project_final.ipynb` and run all cells.
4. In Step 8, enter the patient values when asked (1 = Yes, 0 = No, and the actual age).

## Example
```
Enter Age: 58
Cold_Sweats_Nausea (1 = Yes, 0 = No): 1
...

Heart Disease Prediction : Yes
Patient Risk Score       : 67.66 %
Patient Risk Category    : Medium
```

## Output Folder
Saved models, graphs and the cleaned data are written to the `output` folder.

## Author
Ronit Adhikari
