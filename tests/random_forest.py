import os
import numpy as np

# Define your path
data_path = r'C:\Users\nikna\Documents\pp_datasets\processed_data_1' 
files = [f for f in os.listdir(data_path) if f.endswith('.npy')]

X = []
y = []
triu_idx = np.triu_indices(100, k = 1)

for f in files:
 
    if 'patient' in f:
        matrix = np.load(os.path.join(data_path, f))
        X.append(matrix[triu_idx])
        y.append(1)
        
    elif 'control' in f:
        matrix = np.load(os.path.join(data_path, f))
        X.append(matrix[triu_idx])
        y.append(0)

# Convert to final arrays
X = np.array(X)
y = np.array(y)

print(f"Successfully loaded {len(y)} subjects.")
print(f"Patients: {sum(y)} | Controls: {len(y) - sum(y)}")

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

# 1. Define the Pipeline
# We scale the data (StandardScaler) then run the Forest
pipeline = Pipeline([
    ('scaler', StandardScaler()), 
    ('rf', RandomForestClassifier(n_estimators=500, 
                                  max_features='sqrt', 
                                  random_state=42, 
                                  class_weight='balanced'))
])

# 2. Define Cross-Validation (5-fold is standard for ~200 samples)
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# 3. Run and get multiple metrics
metrics = ['accuracy', 'roc_auc', 'precision', 'recall']
results = cross_validate(pipeline, X, y, cv=cv, scoring=metrics)

# 4. Print Results
print(f"Mean Accuracy: {results['test_accuracy'].mean():.2f}")
print(f"Mean AUC-ROC:  {results['test_roc_auc'].mean():.2f}")
print(f"Mean Recall:   {results['test_recall'].mean():.2f} (Sensitivity to PD)")