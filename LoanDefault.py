# ==============================
# 1. Import Libraries
# ==============================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, roc_auc_score, roc_curve
from sklearn.metrics import ConfusionMatrixDisplay

from imblearn.over_sampling import SMOTE


# ==============================
# 2. Load Dataset
# ==============================

df = pd.read_csv(r"C:\Users\batas\Downloads\archive\Loan_default.csv")

print("First 5 Rows:")
print(df.head())

print("\nDataset Info:")
print(df.info())


# ==============================
# 3. Drop Unnecessary Column
# ==============================

df.drop("LoanID", axis=1, inplace=True)


# ==============================
# 4. Encode Categorical Variables
# ==============================

le = LabelEncoder()
categorical_cols = df.select_dtypes(include='object').columns

for col in categorical_cols:
    df[col] = le.fit_transform(df[col])

print("\nDataset after Encoding:")
print(df.head())


# ==============================
# 5. Separate Features and Target
# ==============================

X = df.drop("Default", axis=1)
y = df["Default"]


# ==============================
# 6. Train-Test Split
# ==============================

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)


# ==============================
# 7. Class Distribution BEFORE SMOTE
# ==============================

plt.figure()
y_train.value_counts().plot(kind='bar')
plt.title("Class Distribution Before SMOTE")
plt.xlabel("Class (0 = Non Default, 1 = Default)")
plt.ylabel("Count")
plt.show()


# ==============================
# 8. Logistic Regression (Baseline)
# ==============================

model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

print("\n===== Baseline Model Results =====")
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))


# Confusion Matrix - Baseline (IMAGE)
cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm,
                              display_labels=["Non Default", "Default"])
disp.plot(cmap="Blues")
plt.title("Confusion Matrix - Baseline Model")
plt.show()


# ==============================
# 9. Apply SMOTE
# ==============================

smote = SMOTE(random_state=42)
X_train_sm, y_train_sm = smote.fit_resample(X_train, y_train)

print("\nBefore SMOTE:\n", y_train.value_counts())
print("After SMOTE:\n", y_train_sm.value_counts())


# ==============================
# 10. Class Distribution AFTER SMOTE
# ==============================

plt.figure()
y_train_sm.value_counts().plot(kind='bar')
plt.title("Class Distribution After SMOTE")
plt.xlabel("Class (0 = Non Default, 1 = Default)")
plt.ylabel("Count")
plt.show()


# ==============================
# 11. Logistic Regression (SMOTE)
# ==============================

model_sm = LogisticRegression(max_iter=1000)
model_sm.fit(X_train_sm, y_train_sm)
y_pred_sm = model_sm.predict(X_test)

print("\n===== SMOTE Model Results =====")
print("Accuracy:", accuracy_score(y_test, y_pred_sm))
print("Classification Report:\n", classification_report(y_test, y_pred_sm))


# Confusion Matrix - SMOTE (IMAGE)
cm_sm = confusion_matrix(y_test, y_pred_sm)
disp_sm = ConfusionMatrixDisplay(confusion_matrix=cm_sm,
                                 display_labels=["Non Default", "Default"])
disp_sm.plot(cmap="Greens")
plt.title("Confusion Matrix - SMOTE Model")
plt.show()


# ==============================
# 12. Sigmoid Function Plot
# ==============================

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

z = np.linspace(-10, 10, 100)
sig = sigmoid(z)

plt.figure()
plt.plot(z, sig)
plt.title("Sigmoid Function")
plt.xlabel("Z value")
plt.ylabel("Sigmoid(z)")
plt.show()


# ==============================
# 13. ROC Curve
# ==============================

y_prob = model_sm.predict_proba(X_test)[:, 1]

fpr, tpr, thresholds = roc_curve(y_test, y_prob)
plt.figure()
plt.plot(fpr, tpr, label="ROC Curve")
plt.plot([0,1], [0,1], 'k--')
plt.title("ROC Curve - SMOTE Model")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.legend()
plt.show()

print("AUC Score:", roc_auc_score(y_test, y_prob))