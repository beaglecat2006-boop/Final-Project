# Final-Project
Final Project usage repository

Final Project Code

import numpy as np import pandas as pd

def convert_target(x): if x > 0: return 1 else: return 0

def clean_heart_data(df: pd.DataFrame) -> pd.DataFrame: cleaned_df = df.copy()

cleaned_df["ca"] = cleaned_df["ca"].fillna(0.0)
cleaned_df["thal"] = cleaned_df["thal"].fillna(3.0)
cleaned_df["target"] = cleaned_df["target"].apply(convert_target)

return cleaned_df
import pandas as pd

column_names = [ "age", "sex", "cp", "trestbps", "chol", "fbs", "restecg", "thalach", "exang", "oldpeak", "slope", "ca", "thal", "target", ]

raw_df = pd.read_csv( "https://uci.edu", names=column_names, na_values="?", ) cleaned_df = clean_heart_data(raw_df) cleaned_df.to_csv("cleaned_heart.csv")

print("Notebook 1 Complete: 'cleaned_heart.csv' created.")

from sklearn.model_selection import train_test_split import pandas as pd

df = pd.read_csv("cleaned_heart.csv")

X = df.drop("target", axis=1) y = df["target"]

X_train, X_test, y_train, y_test = train_test_split( X, y, test_size=0.2, random_state=42, stratify=y )

import matplotlib.pyplot as plt import seaborn as sns

sns.set_theme(style="whitegrid")

plt.figure(figsize=(6, 4)) sns.boxplot(x="target", y="chol", data=df) plt.title("Cholesterol Levels by Heart Disease Status") plt.show()

plt.figure(figsize=(10, 6)) sns.heatmap(df.corr(), annot=True, cmap="coolwarm", fmt=".2f") plt.title("Clinical Correlation Map") plt.show()

plt.figure(figsize=(6, 4)) sns.countplot(x="cp", hue="target", data=df) plt.title("Chest Pain Frequency by Diagnosis") plt.show()

import pandas as pd from sklearn.linear_model import LogisticRegression from sklearn.svm import SVC

log_reg = LogisticRegression() log_reg.fit(X_train, y_train)

svm_model = SVC(kernel="rbf") svm_model.fit(X_train, y_train)

print("Logistic Regression Results:") print("Train Score:", log_reg.score(X_train, y_train)) print("Test Score:", log_reg.score(X_test, y_test))

print("\nSVM Results:") print("Train Score:", svm_model.score(X_train, y_train)) print("Test Score:", svm_model.score(X_test, y_test))
