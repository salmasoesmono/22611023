import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, KFold, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.preprocessing import LabelEncoder, StandardScaler

# Load and preprocess data
data = pd.read_csv(r'D:\SEMESTER 4\MPML\UAS\insurance.csv')

# Discretize 'age' into categories
bins = [0, 20, 30, 40, 50, 60, 70, 80, np.inf]
labels = ['<20', '20-30', '30-40', '40-50', '50-60', '60-70', '70-80', '80+']
data['age_category'] = pd.cut(data['age'], bins=bins, labels=labels, right=False)

# Define features and target variable
X = data.drop(["age", "age_category"], axis=1)  # Features
y = data["age_category"]  # Target variable

# Encode categorical variables
categorical_columns = ['sex', 'smoker', 'region']
label_encoders = {}

for column in categorical_columns:
    if column in X.columns:
        le = LabelEncoder()
        X[column] = le.fit_transform(X[column].astype(str))
        label_encoders[column] = le

# Normalize numerical features
scaler = StandardScaler()
X[['children', 'charges']] = scaler.fit_transform(X[['children', 'charges']])

# Check the data types and unique values to ensure all data is numeric
print("Data types of features:\n", X.dtypes)
print("Unique values in each feature:\n", X.nunique())

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define models and hyperparameter grids for classification
models = {
    "Random Forest": RandomForestClassifier(random_state=42),
    "KNN": KNeighborsClassifier(),
    "Decision Tree": DecisionTreeClassifier(random_state=42)
}

param_grids = {
    "Random Forest": {
        "n_estimators": [100, 200, 300],
        "max_depth": [5, 10, 15],
    },
    "KNN": {
        "n_neighbors": [3, 5, 7, 9],
        "weights": ['uniform', 'distance'],
        "p": [1, 2]  # 1 untuk Manhattan, 2 untuk Euclidean
    },
    "Decision Tree": {
        "max_depth": [5, 10, 15],
        "min_samples_split": [2, 5, 10]
    }
}

# K-Fold Cross-Validation for Robust Evaluation
kfold = KFold(n_splits=5, shuffle=True, random_state=42)
results = {}

for model_name, model in models.items():
    print(f"Evaluating {model_name}...")
    cv_results = []
    
    # Hyperparameter tuning with GridSearchCV
    if model_name in param_grids:
        grid_search = GridSearchCV(model, param_grids[model_name], scoring='accuracy', cv=kfold)
        grid_search.fit(X_train, y_train)
        best_model = grid_search.best_estimator_
        best_params = grid_search.best_params_
        best_score = grid_search.best_score_
        
        print(f"Best Parameters for {model_name}: {best_params}")
        print(f"Best Score for {model_name}: {best_score:.4f}")

    else:
        best_model = model
        best_params = {}
        best_score = np.nan  # No grid search for models without hyperparameters

    for train_index, val_index in kfold.split(X_train):
        X_train_fold, X_val_fold = X_train.iloc[train_index], X_train.iloc[val_index]
        y_train_fold, y_val_fold = y_train.iloc[train_index], y_train.iloc[val_index]

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
