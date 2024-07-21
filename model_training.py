import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, KFold, GridSearchCV
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.preprocessing import LabelEncoder, OrdinalEncoder

# Load and preprocess data
data = pd.read_csv(r'D:\SEMESTER 4\MPML\UAS\onlinefoods.csv')

# Define features and target variable
X = data.drop("Feedback", axis=1)  # Features
y = data["Feedback"]  # Target variable

# Encode categorical variables
categorical_columns = ['Gender', 'Marital Status', 'Occupation', 'Educational Qualifications', 'Output2']
label_encoders = {}

for column in categorical_columns:
    if column in X.columns:
        le = LabelEncoder()
        X[column] = le.fit_transform(X[column].astype(str))
        label_encoders[column] = le

# Convert 'Monthly Income' and 'Output' using OrdinalEncoder for more flexible handling
ordinal_columns = ['Monthly Income', 'Output']
ordinal_encoder = OrdinalEncoder()

X[ordinal_columns] = ordinal_encoder.fit_transform(X[ordinal_columns].astype(str))

# Encode target variable if it's not already numerical
if y.dtype == 'object':
    le_feedback = LabelEncoder()
    y = le_feedback.fit_transform(y)

# Check the data types and unique values to ensure all data is numeric
print("Data types of features:\n", X.dtypes)
print("Unique values in each feature:\n", X.nunique())

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define models and hyperparameter grids
models = {
    "Random Forest": RandomForestClassifier(random_state=42),
    "Gradient Boosting": GradientBoostingClassifier(random_state=42),
    "SVM": SVC(kernel="rbf", random_state=42),
}

param_grids = {
    "Random Forest": {
        "n_estimators": [100, 200, 300],
        "max_depth": [5, 10, 15],
    },
    "Gradient Boosting": {"learning_rate": [0.1, 0.05, 0.01], "n_estimators": [100, 200, 300]},
    "SVM": {"C": [0.1, 1, 10], "gamma": [0.01, 0.1, 1]},
}

# K-Fold Cross-Validation for Robust Evaluation
kfold = KFold(n_splits=5, shuffle=True, random_state=42)
results = {}

for model_name, model in models.items():
    print(f"Evaluating {model_name}...")
    cv_results = []
    
    # Hyperparameter tuning with GridSearchCV
    grid_search = GridSearchCV(model, param_grids[model_name], scoring='accuracy', cv=kfold)
    grid_search.fit(X_train, y_train)
    best_model = grid_search.best_estimator_
    best_params = grid_search.best_params_
    best_score = grid_search.best_score_
    
    print(f"Best Parameters for {model_name}: {best_params}")
    print(f"Best Score for {model_name}: {best_score:.4f}")

    for train_index, val_index in kfold.split(X_train):
        X_train_fold, X_val_fold = X_train.iloc[train_index], X_train.iloc[val_index]
        y_train_fold, y_val_fold = y_train[train_index], y_train[val_index]

        best_model.fit(X_train_fold, y_train_fold)  # Train the best model

        y_pred = best_model.predict(X_val_fold)
        accuracy = accuracy_score(y_val_fold, y_pred)
        precision = precision_score(y_val_fold, y_pred, average='weighted')
        recall = recall_score(y_val_fold, y_pred, average='weighted')
        f1 = f1_score(y_val_fold, y_pred, average='weighted')
        cv_results.append((accuracy, precision, recall, f1))

    # Calculate mean and standard deviation for cross-validation metrics
    mean_cv_results = np.mean(cv_results, axis=0)
    std_cv_results = np.std(cv_results, axis=0)
    
    # Record performance metrics
    results[model_name] = {
        "Mean Accuracy": mean_cv_results[0],
        "Mean Precision": mean_cv_results[1],
        "Mean Recall": mean_cv_results[2],
        "Mean F1-Score": mean_cv_results[3],
        "Std Accuracy": std_cv_results[0],
        "Std Precision": std_cv_results[1],
        "Std Recall": std_cv_results[2],
        "Std F1-Score": std_cv_results[3],
    }

# Print out the performance metrics for all models
print("\nPerformance Metrics:")
for model_name, metrics in results.items():
    print(f"\n{model_name}:")
    for metric_name, value in metrics.items():
        print(f"{metric_name}: {value:.4f}")

# Visualize the results
results_df = pd.DataFrame(results).T

# Plotting the metrics
fig, ax = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle('Model Performance Metrics', fontsize=16)

sns.barplot(x=results_df.index, y='Mean Accuracy', data=results_df, ax=ax[0, 0])
ax[0, 0].set_title('Mean Accuracy')
ax[0, 0].set_ylabel('Mean Accuracy')

sns.barplot(x=results_df.index, y='Mean Precision', data=results_df, ax=ax[0, 1])
ax[0, 1].set_title('Mean Precision')
ax[0, 1].set_ylabel('Mean Precision')

sns.barplot(x=results_df.index, y='Mean Recall', data=results_df, ax=ax[1, 0])
ax[1, 0].set_title('Mean Recall')
ax[1, 0].set_ylabel('Mean Recall')

sns.barplot(x=results_df.index, y='Mean F1-Score', data=results_df, ax=ax[1, 1])
ax[1, 1].set_title('Mean F1-Score')
ax[1, 1].set_ylabel('Mean F1-Score')

plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.show()
