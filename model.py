import pandas as pd
import numpy as np
import pickle

from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split,cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

df = pd.read_csv("Impact_of_Remote_Work_on_Mental_Health.csv")

df = df.drop(columns=['Employee_ID'],axis=1)

df['target'] = df['Satisfaction_with_Remote_Work'].apply(lambda x: 1 if x == 'Satisfied' else 2 if x == "Unsatisfied" else 0)

y = df['target']
x = df.drop(columns=['Satisfaction_with_Remote_Work','target'],axis=1)

x_cat = x.select_dtypes(include='object')
x_num = x.select_dtypes(exclude='object')

preprocess = ColumnTransformer(
    transformers=(
        ('num',Pipeline(steps=[
            ('impute', SimpleImputer(strategy='median')),
            ('scaler', MinMaxScaler())
        ]),x_num.columns),
        ('cat',Pipeline(steps=[
            ('impute', SimpleImputer(strategy='most_frequent')),
            ('encode',OneHotEncoder(handle_unknown='ignore'))
        ]),x_cat.columns)
    )
)

pipeline =Pipeline(steps=[
    ('preprocess', preprocess),
    ('classifier', RandomForestClassifier(n_estimators=100, random_state=42))
])

scores = cross_val_score(pipeline, x, y, cv=5, scoring='accuracy')
print("Cross-Validation Accuracy Scores:", scores)
print("Mean Accuracy:", scores.mean())

x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state=42)

pipeline.fit(x_train,y_train)

y_pred = pipeline.predict(x_test)

# print(accuracy_score(y_test,y_pred))
# print("\nClassification Report:\n", classification_report(y_test, y_pred))
# print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))

# Save the trained model using pickle
path = "./model/remote_work_model.pkl"

with open(path, "wb") as f:
    pickle.dump(pipeline, f)

# Load the model (optional, for demonstration)
with open(path, "rb") as f:
    loaded_model = pickle.load(f)

# Example of using the loaded model for prediction
loaded_pred = loaded_model.predict(x_test)
print("\nLoaded Model Test Accuracy:", accuracy_score(y_test, loaded_pred))