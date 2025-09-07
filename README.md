# Assignment 1 - Predicting Heart Disease
- for assignment 1, I want to predict whether a patient is likely to have heart disease based on their [medical and demographic data](https://www.kaggle.com/datasets/johnsmith88/heart-disease-dataset)
- target variable: presence of heart disease (1 = yes, 2 = no)

## Features
- age
- gender
- chest pain type
- resting blood pressure
- serum cholestrol
- fasting blood pressure
- resting electrocardiographic results
- maximum heart rate achieved
- exercise induced angina
- oldpeak
- slope of the peak exercise st segment
- number of major vessels colored by flouroscopy
- thal: 0 = normal; 1 = fixed defect; 2 = reversable defect

## Code
we will be using [johnsmith88's heart disease dataset](https://www.kaggle.com/datasets/johnsmith88/heart-disease-dataset) which we downloaded locally into our directory
```python
data = pd.read_csv('heart.csv')
```

we then binarize targets to check for potential multi-class targets (0-4 in some heart disease datasets), and if present binarizes the target (0 = no, 1 = yes) to match our binary classification goal
```python
if data['target'].max() > 1:
    data['target'] = data['target'].apply(lambda x: 0 if x == 0 else 1)
```

we define the features and targets to focus on relevant predictors of heart disease based on medical significance
```python
features = ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 'thalach',
            'exang', 'oldpeak', 'slope', 'ca', 'thal']
X = data[features]
y = data['target']
```

we do preprocessing on the data with one-hot encoding which converts categorical variables (cp, restecg, slope, thal) into binary columns (e.g., cp_1, cp_2) because machine learning models require numerical input
and standardization to normalize numerical features to prevent features with larger ranges (e.g., chol) from dominating the model, ensuring fair contribution from all predictors
```python
categorical_cols = ['cp', 'restecg', 'slope', 'thal']
X = pd.get_dummies(X, columns=categorical_cols, drop_first=True)

scaler = StandardScaler()
numerical_cols = ['age', 'trestbps', 'chol', 'thalach', 'oldpeak', 'ca']
X[numerical_cols] = scaler.fit_transform(X[numerical_cols])
```

we then train the models with LogicalRegression and RandomForestClassifier which are fitted to the data to learn patterns with a 80/20 split to balance training depth and test evaluation
```python
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

log_reg = LogisticRegression(random_state=42)
rf_clf = RandomForestClassifier(random_state=42)
log_reg.fit(X_train, y_train)
rf_clf.fit(X_train, y_train)
```

finally we evaluate the results for a detailed breakdown of performance per class and roc_auc_score to measure the models’ ability to distinguish between classes, providing a comprehensive assessment of predictive power
```python
print("\nlogistic regression:")
print(classification_report(y_test, log_reg.predict(X_test)))
print("ROC-AUC:", roc_auc_score(y_test, log_reg.predict_proba(X_test)[:, 1]))
print("\nrandom forest:")
print(classification_report(y_test, rf_clf.predict(X_test)))
print("ROC-AUC:", roc_auc_score(y_test, rf_clf.predict_proba(X_test)[:, 1]))
```


## Analysis
- the dataset provides us with 499 instances of no heart disease, and 526 instances with heart disease found
- we use the dataset with classification models to identify patterns with the given features that correlate with heart disease diagnosis, using supervised machine learning techniques
- the classification models we use are: logistic regression and random forest classifier
  - we use logistic regression for the simplicity and interpretability. it assumes a linear relationship between the features and the log odds of heart disease, making it suitable initial baseline performance and understanding feature importance.
  - random forest classifier is used for its whole approach, which combines multiple decision trees to reduce overfitting and capture nonlinear relationships as it offers a higher accuracy potential
- both models are trained with a 80/20 train-test split, with preprocessing including one-hot encoding for categorical variables (cp, restecg, slope, thal) and standardization for numerical variables (age, trestbps, chol, thalach, oldpeak, ca)

### Model evaluation
- models were evaluated on a test set of 205 instances using precision, recall, f1-score, accuracy, and roc-auc metrics
  - logistic regression
    - accuracy: 0.82
    - precision: 0.87 (class 0), 0.78 (class 1)
    - recall: 0.75 (class 0), 0.89 (class 1)
    - f1-score: 0.80 (class 0), 0.83 (class 1)
    - roc-auc: 0.886
    - we can derive that the performance indicates a good balance, with a strong recall for class 1 (heart disease), which suggests effective identification of positive cases, with still some false positives that reduce the precision
  - random forest
    - accuracy: 1.00
    - precision: 1.00 (class 0), 1.00 (class 1)
    - recall: 1.00 (class 0), 1.00 (class 1)
    - f1-score: 1.00 (class 0), 1.00 (class 1)
    - roc-auc: 1.00
    - we see using random forest shows perfect classification, which is likely due to the classification model's ability to capture more complex patterns and the structure of the dataset
- we find that using random forest classifier significantly outperforms logistic regression, with it having a perfect classification rate while logisitic regression provides a solid baseline with an 82% accuracy and high roc-auc which still indicates good discriminative power
- overall, through the results I would prefer random forest as my choice of classification model due to its superior performance. Logsitic regression was good for its interpretability but may cause us to miss nuances in the data, where in a high stake field regarding medical precision can be fatal. 
