# Importing libraries
from __future__ import print_function
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report
from sklearn import metrics
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
import pickle
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from sklearn import metrics
from sklearn.preprocessing import LabelEncoder




# Load the dataset
PATH = r'E:\crop\Crop_recommendation (1).csv'
df = pd.read_csv(PATH)

# Display basic information about the dataset
print(df.head())
print(df.info())
label_encoder = LabelEncoder()
df['label'] = label_encoder.fit_transform(df['label'])


# Identify unique classes in the 'label' column
unique_classes = df['label'].unique()

# Print the unique classes
print("Unique Classes:", unique_classes)

# Encode the target variable using Label Encoder
label_encoder = LabelEncoder()
df['label'] = label_encoder.fit_transform(df['label'])

# Split the dataset into training and testing sets
features = df.drop('label', axis=1)
labels = df['label']
Xtrain, Xtest, Ytrain, Ytest = train_test_split(features, labels, test_size=0.2, random_state=42)


# Create and train the XGBoost model
XB = XGBClassifier()
XB.fit(Xtrain, Ytrain)


# Evaluate the model
predicted_values = XB.predict(Xtest)
accuracy = metrics.accuracy_score(Ytest, predicted_values)
print("XGBoost's Accuracy is: ", accuracy * 100)

# Decision Tree
DecisionTree = DecisionTreeClassifier(criterion="entropy", random_state=2, max_depth=5)
DecisionTree.fit(Xtrain, Ytrain)
predicted_values = DecisionTree.predict(Xtest)
accuracy_dt = metrics.accuracy_score(Ytest, predicted_values)
print("Decision Tree's Accuracy is:", accuracy_dt * 100)

# Cross-validation score (Decision Tree)
score_dt = cross_val_score(DecisionTree, features, labels, cv=5)
print("Cross-validation scores (Decision Tree):", score_dt)



# Save the Decision Tree model
with open('DecisionTree.pkl', 'wb') as model_pkl:
    pickle.dump(DecisionTree, model_pkl)

# Naive Bayes
NaiveBayes = GaussianNB()
NaiveBayes.fit(Xtrain, Ytrain)
predicted_values_nb = NaiveBayes.predict(Xtest)
accuracy_nb = metrics.accuracy_score(Ytest, predicted_values_nb)
print("Naive Bayes's Accuracy is:", accuracy_nb * 100)

# Cross-validation score (Naive Bayes)
score_nb = cross_val_score(NaiveBayes, features, labels, cv=5)
print("Cross-validation scores (Naive Bayes):", score_nb)


# Save the Naive Bayes model
with open('NaiveBayes.pkl', 'wb') as model_pkl:
    pickle.dump(NaiveBayes, model_pkl)

# SVM
SVM = SVC(gamma='auto')
SVM.fit(Xtrain, Ytrain)
predicted_values_svm = SVM.predict(Xtest)
accuracy_svm = metrics.accuracy_score(Ytest, predicted_values_svm)
print("SVM's Accuracy is:", accuracy_svm * 100)

# Cross-validation score (SVM)
score_svm = cross_val_score(SVM, features, labels, cv=5)
print("Cross-validation scores (SVM):", score_svm)

# Save the SVM model
with open('SVM.pkl', 'wb') as model_pkl:
    pickle.dump(SVM, model_pkl)

# Logistic Regression
LogReg = LogisticRegression(random_state=2)
LogReg.fit(Xtrain, Ytrain)
predicted_values_lr = LogReg.predict(Xtest)
accuracy_lr = metrics.accuracy_score(Ytest, predicted_values_lr)
print("Logistic Regression's Accuracy is:", accuracy_lr * 100)

# Cross-validation score (Logistic Regression)
score_lr = cross_val_score(LogReg, features, labels, cv=5)
print("Cross-validation scores (Logistic Regression):", score_lr)


# Save the Logistic Regression model
with open('LogisticRegression.pkl', 'wb') as model_pkl:
    pickle.dump(LogReg, model_pkl)

# Random Forest
RF = RandomForestClassifier(n_estimators=20, random_state=0)
RF.fit(Xtrain, Ytrain)
predicted_values_rf = RF.predict(Xtest)
accuracy_rf = metrics.accuracy_score(Ytest, predicted_values_rf)
print("Random Forest's Accuracy is:", accuracy_rf * 100)

# Cross-validation score (Random Forest)
score_rf = cross_val_score(RF, features, labels, cv=5)
print("Cross-validation scores (Random Forest):", score_rf)


# Save the Random Forest model
with open('RandomForest.pkl', 'wb') as model_pkl:
    pickle.dump(RF, model_pkl)

# XGBoost
XB = XGBClassifier()
XB.fit(Xtrain, Ytrain)
predicted_values_xb = XB.predict(Xtest)
accuracy_xb = metrics.accuracy_score(Ytest, predicted_values_xb)
print("XGBoost's Accuracy is:", accuracy_xb * 100)

# Cross-validation score (XGBoost)
score_xb = cross_val_score(XB, features, labels, cv=5)
print("Cross-validation scores (XGBoost):", score_xb)


# Save the XGBoost model
with open('XGBoost.pkl', 'wb') as model_pkl:
    pickle.dump(XB, model_pkl)

# Plotting the accuracy comparison
models = ['Decision Tree', 'Naive Bayes', 'SVM', 'Logistic Regression', 'Random Forest', 'XGBoost']
accuracies = [accuracy_dt, accuracy_nb, accuracy_svm, accuracy_lr, accuracy_rf, accuracy_xb]

plt.figure(figsize=[10, 5], dpi=100)
plt.title('Accuracy Comparison')
plt.xlabel('Accuracy')
plt.ylabel('Algorithm')
sns.barplot(x=accuracies, y=models, palette='dark')
plt.show()

# Display accuracy for each model
accuracy_models = dict(zip(models, accuracies))
for model, acc in accuracy_models.items():
    print(f"{model}'s Accuracy is: {acc * 100}")

# Example predictions
data_example1 = np.array([[104, 18, 30, 23.603016, 60.3, 6.7, 140.91]])
prediction_example1 = RF.predict(data_example1)
print("Example 1 Prediction:", prediction_example1)

data_example2 = np.array([[83, 45, 60, 28, 70.3, 7.0, 150.9]])
prediction_example2 = RF.predict(data_example2)
print("Example 2 Prediction:", prediction_example2)
