import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, roc_auc_score

# load dataset
data = pd.read_csv('heart.csv')

# let's check the column names and target values
print("column names:", data.columns)
print("target value counts:", data['target'].value_counts())

# binarize target if necessary (if values are 0–4; skip if already 0/1)
if data['target'].max() > 1:
    data['target'] = data['target'].apply(lambda x: 0 if x == 0 else 1)

# define our features and target
features = ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 'thalach',
            'exang', 'oldpeak', 'slope', 'ca', 'thal']
X = data[features]
y = data['target']

# check for missing values
print("missing values:\n", X.isnull().sum())

# categorical variables
categorical_cols = ['cp', 'restecg', 'slope', 'thal']
X = pd.get_dummies(X, columns=categorical_cols, drop_first=True)

# scale numerical features
scaler = StandardScaler()
numerical_cols = ['age', 'trestbps', 'chol', 'thalach', 'oldpeak', 'ca']
X[numerical_cols] = scaler.fit_transform(X[numerical_cols])

# split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# train models
log_reg = LogisticRegression(random_state=42)
rf_clf = RandomForestClassifier(random_state=42)
log_reg.fit(X_train, y_train)
rf_clf.fit(X_train, y_train)

# evaluate models
print("\nlogistic regression:")
print(classification_report(y_test, log_reg.predict(X_test)))
print("ROC-AUC:", roc_auc_score(y_test, log_reg.predict_proba(X_test)[:, 1]))
print("\nrandom forest:")
print(classification_report(y_test, rf_clf.predict(X_test)))
print("ROC-AUC:", roc_auc_score(y_test, rf_clf.predict_proba(X_test)[:, 1]))