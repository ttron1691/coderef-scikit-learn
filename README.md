# Code Reference for scikit-learn
## Install Packages
```Bash
pip install -U scikit-learn
```
## Model Selection
In this section we summarize the methods concerning model selection.
### Cross-Validation
In order to split the input data into a training and test data set we can use the following method
```Python
sklearn.model_selection.train_test_split(*arrays, test_size=None, train_size=None, random_state=None, shuffle=True, stratify=None)
```
Example:
```Python
from sklearn.model_selection import train_test_split
from sklearn import datasets
from sklearn import svm

X, y = datasets.load_iris(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=0)
```
### Tuning hyper-parameters of an estimator
### Metrics and scoring
### Validation curves
```Python

```
## Linear models
### Linear regression
The most general linear model expression is given by
$$\hat y=\theta_0+\theta_1x_1+\dots+\theta_nx_n=\theta_0+\sum_{i=1}^n\theta_ix_i=\mathbf{\theta}\cdot\mathbf{X}$$
with $\mathbf{\theta}=(\theta_0,\theta_1,\dots,\theta_n)^T$ and $\mathbf{X}=(x_0,x_1,\dots,x_n)^T$ and $x_0=1$.
```Python
from sklearn import linear_model
reg = linear_model.LinearRegression()
reg.fit([[0, 0], [1, 1], [2, 2]], [0, 1, 2])
reg.coef_
```

# Scikit-learn Reference Card

## Table of Contents
1. [Data Preparation](#data-preparation)
2. [Model Selection](#model-selection)
3. [Model Training & Evaluation](#model-training--evaluation)
4. [Hyperparameter Tuning](#hyperparameter-tuning)
5. [Feature Engineering](#feature-engineering)
6. [Pipeline Construction](#pipeline-construction)
7. [Model Persistence](#model-persistence)
8. [Common Algorithms](#common-algorithms)

## Data Preparation

### Loading Data
```python
# From CSV file
import pandas as pd
df = pd.read_csv('data.csv')
X = df.drop('target', axis=1)
y = df['target']

# From scikit-learn datasets
from sklearn import datasets
iris = datasets.load_iris()
X, y = iris.data, iris.target
```

### Splitting Data
```python
from sklearn.model_selection import train_test_split

# Basic split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)

# Stratified split (maintains class distribution)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42)
```

### Handling Missing Values
```python
from sklearn.impute import SimpleImputer

# Replace missing values with mean
imputer = SimpleImputer(strategy='mean')
X_imputed = imputer.fit_transform(X)

# Other strategies: 'median', 'most_frequent', 'constant'
```

### Scaling Features
```python
from sklearn.preprocessing import StandardScaler, MinMaxScaler

# Standardization (mean=0, std=1)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Normalization (range 0-1)
normalizer = MinMaxScaler()
X_normalized = normalizer.fit_transform(X)
```

### Encoding Categorical Features
```python
from sklearn.preprocessing import OneHotEncoder, LabelEncoder

# For target variable
le = LabelEncoder()
y_encoded = le.fit_transform(y)

# For feature variables
encoder = OneHotEncoder(sparse_output=False)
X_encoded = encoder.fit_transform(X[['categorical_feature']])
```

## Model Selection

### Classification Algorithms
```python
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB

# Quick model comparison
models = {
    'Logistic Regression': LogisticRegression(),
    'Decision Tree': DecisionTreeClassifier(),
    'Random Forest': RandomForestClassifier(),
    'Gradient Boosting': GradientBoostingClassifier(),
    'SVM': SVC(probability=True),
    'KNN': KNeighborsClassifier(),
    'Naive Bayes': GaussianNB()
}

for name, model in models.items():
    model.fit(X_train, y_train)
    print(f"{name} accuracy: {model.score(X_test, y_test):.4f}")
```

### Regression Algorithms
```python
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR

# Quick model comparison
reg_models = {
    'Linear Regression': LinearRegression(),
    'Ridge': Ridge(),
    'Lasso': Lasso(),
    'ElasticNet': ElasticNet(),
    'Decision Tree': DecisionTreeRegressor(),
    'Random Forest': RandomForestRegressor(),
    'Gradient Boosting': GradientBoostingRegressor(),
    'SVR': SVR()
}
```

## Model Training & Evaluation

### Basic Training and Prediction
```python
# Training
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

# Prediction
y_pred = model.predict(X_test)
y_pred_proba = model.predict_proba(X_test)  # For classifiers
```

### Classification Metrics
```python
from sklearn.metrics import (accuracy_score, precision_score, recall_score, 
                            f1_score, classification_report, confusion_matrix,
                            roc_auc_score, roc_curve)

# Basic metrics
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average='weighted')
recall = recall_score(y_test, y_pred, average='weighted')
f1 = f1_score(y_test, y_pred, average='weighted')

# Comprehensive report
print(classification_report(y_test, y_pred))

# Confusion matrix
import matplotlib.pyplot as plt
import seaborn as sns

cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()

# ROC curve for binary classification
fpr, tpr, _ = roc_curve(y_test, y_pred_proba[:, 1])
auc = roc_auc_score(y_test, y_pred_proba[:, 1])
plt.plot(fpr, tpr, label=f'AUC = {auc:.3f}')
plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend()
plt.show()
```

### Regression Metrics
```python
from sklearn.metrics import (mean_squared_error, mean_absolute_error, 
                           r2_score, explained_variance_score)

# Calculate metrics
mse = mean_squared_error(y_test, y_pred)
rmse = mean_squared_error(y_test, y_pred, squared=False)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
ev = explained_variance_score(y_test, y_pred)

print(f"MSE: {mse:.4f}")
print(f"RMSE: {rmse:.4f}")
print(f"MAE: {mae:.4f}")
print(f"R²: {r2:.4f}")
print(f"Explained Variance: {ev:.4f}")
```

### Cross-Validation
```python
from sklearn.model_selection import cross_val_score, KFold, StratifiedKFold

# Basic cross-validation
cv_scores = cross_val_score(model, X, y, cv=5)
print(f"CV Scores: {cv_scores}")
print(f"Mean CV Score: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

# Custom cross-validation
kfold = KFold(n_splits=5, shuffle=True, random_state=42)
cv_scores = cross_val_score(model, X, y, cv=kfold, scoring='accuracy')

# Stratified cross-validation (for classification)
stratified_kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cv_scores = cross_val_score(model, X, y, cv=stratified_kfold)
```

## Hyperparameter Tuning

### Grid Search
```python
from sklearn.model_selection import GridSearchCV

# Define parameter grid
param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [None, 10, 20, 30],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

# Perform grid search
grid_search = GridSearchCV(
    RandomForestClassifier(random_state=42),
    param_grid=param_grid,
    cv=5,
    scoring='accuracy',
    n_jobs=-1,
    verbose=1
)
grid_search.fit(X_train, y_train)

# Best parameters and score
print(f"Best parameters: {grid_search.best_params_}")
print(f"Best cross-validation score: {grid_search.best_score_:.4f}")

# Get best model
best_model = grid_search.best_estimator_
```

### RandomizedSearchCV
```python
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import randint, uniform

# Define parameter distributions
param_dist = {
    'n_estimators': randint(50, 500),
    'max_depth': randint(5, 50),
    'min_samples_split': randint(2, 20),
    'min_samples_leaf': randint(1, 10),
    'max_features': uniform(0.1, 0.9)
}

# Perform randomized search
random_search = RandomizedSearchCV(
    RandomForestClassifier(random_state=42),
    param_distributions=param_dist,
    n_iter=100,
    cv=5,
    scoring='accuracy',
    n_jobs=-1,
    random_state=42,
    verbose=1
)
random_search.fit(X_train, y_train)

# Best parameters and score
print(f"Best parameters: {random_search.best_params_}")
print(f"Best cross-validation score: {random_search.best_score_:.4f}")
```

## Feature Engineering

### Feature Selection
```python
from sklearn.feature_selection import (SelectKBest, f_classif, RFE,
                                      SelectFromModel)

# Select k best features
selector = SelectKBest(f_classif, k=10)
X_new = selector.fit_transform(X, y)
selected_indices = selector.get_support(indices=True)
selected_features = X.columns[selected_indices]

# Recursive Feature Elimination
from sklearn.linear_model import LogisticRegression
rfe = RFE(estimator=LogisticRegression(), n_features_to_select=10)
X_rfe = rfe.fit_transform(X, y)
selected_features = X.columns[rfe.support_]

# Feature selection with model
selector = SelectFromModel(RandomForestClassifier(random_state=42), threshold='median')
X_sfm = selector.fit_transform(X, y)
selected_features = X.columns[selector.get_support()]
```

### Polynomial Features
```python
from sklearn.preprocessing import PolynomialFeatures

# Generate polynomial features
poly = PolynomialFeatures(degree=2, include_bias=False)
X_poly = poly.fit_transform(X)

# See feature names
if isinstance(X, pd.DataFrame):
    feature_names = poly.get_feature_names_out(X.columns)
    print(feature_names)
```

## Pipeline Construction

### Basic Pipeline
```python
from sklearn.pipeline import Pipeline

# Classification pipeline
pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('classifier', RandomForestClassifier(random_state=42))
])

# Fit and predict
pipeline.fit(X_train, y_train)
y_pred = pipeline.predict(X_test)
```

### Advanced Pipeline with ColumnTransformer
```python
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer

# Define column types
numerical_cols = ['age', 'income', 'score']
categorical_cols = ['gender', 'country', 'education']

# Define preprocessors
numerical_transformer = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

categorical_transformer = Pipeline([
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('encoder', OneHotEncoder(handle_unknown='ignore'))
])

# Create preprocessor
preprocessor = ColumnTransformer([
    ('num', numerical_transformer, numerical_cols),
    ('cat', categorical_transformer, categorical_cols)
])

# Create full pipeline
full_pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('classifier', RandomForestClassifier(random_state=42))
])

# Fit and predict
full_pipeline.fit(X_train, y_train)
y_pred = full_pipeline.predict(X_test)
```

### Pipeline with Parameter Tuning
```python
# Define pipeline
pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('classifier', RandomForestClassifier(random_state=42))
])

# Parameter grid for pipeline components
param_grid = {
    'classifier__n_estimators': [100, 200],
    'classifier__max_depth': [None, 10, 20],
    'classifier__min_samples_split': [2, 5]
}

# Grid search with pipeline
grid_search = GridSearchCV(
    pipeline,
    param_grid=param_grid,
    cv=5,
    n_jobs=-1
)
grid_search.fit(X_train, y_train)
```

## Model Persistence

### Saving and Loading Models
```python
import joblib
import pickle

# Using joblib (recommended for large numpy arrays)
joblib.dump(model, 'model.joblib')
loaded_model = joblib.load('model.joblib')

# Using pickle
with open('model.pkl', 'wb') as file:
    pickle.dump(model, file)
    
with open('model.pkl', 'rb') as file:
    loaded_model = pickle.load(file)

# Saving pipelines
joblib.dump(pipeline, 'pipeline.joblib')
loaded_pipeline = joblib.load('pipeline.joblib')
```

## Common Algorithms

### Linear Models
```python
from sklearn.linear_model import (LinearRegression, Ridge, Lasso, ElasticNet,
                                LogisticRegression, SGDClassifier)

# Linear regression
lr = LinearRegression()
lr.fit(X_train, y_train)
print(f"Coefficients: {lr.coef_}")
print(f"Intercept: {lr.intercept_}")

# Regularized regression
ridge = Ridge(alpha=1.0)
lasso = Lasso(alpha=0.1)
elastic = ElasticNet(alpha=0.1, l1_ratio=0.5)

# Logistic regression
logistic = LogisticRegression(C=1.0, penalty='l2', solver='liblinear')
logistic.fit(X_train, y_train)
```

### Tree-Based Models
```python
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import (RandomForestClassifier, RandomForestRegressor,
                             GradientBoostingClassifier, GradientBoostingRegressor,
                             AdaBoostClassifier, AdaBoostRegressor)

# Decision tree with visualization
tree_model = DecisionTreeClassifier(max_depth=3)
tree_model.fit(X_train, y_train)

# Visualize tree
from sklearn.tree import plot_tree
plt.figure(figsize=(20, 10))
plot_tree(tree_model, filled=True, feature_names=X.columns if hasattr(X, 'columns') else None, 
          class_names=[str(c) for c in tree_model.classes_], rounded=True)
plt.show()

# Random Forest
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)

# Feature importances
importances = rf.feature_importances_
if hasattr(X, 'columns'):
    for feature, importance in zip(X.columns, importances):
        print(f"{feature}: {importance:.4f}")

# Gradient Boosting
gb = GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, random_state=42)
gb.fit(X_train, y_train)
```

### Support Vector Machines
```python
from sklearn.svm import SVC, SVR

# SVM Classifier with different kernels
svc_linear = SVC(kernel='linear', C=1.0, random_state=42)
svc_rbf = SVC(kernel='rbf', C=1.0, gamma='scale', random_state=42)
svc_poly = SVC(kernel='poly', degree=3, C=1.0, random_state=42)

# SVM Regressor
svr = SVR(kernel='rbf', C=1.0, epsilon=0.1)
```

### Clustering
```python
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.mixture import GaussianMixture

# K-means clustering
kmeans = KMeans(n_clusters=3, random_state=42)
clusters = kmeans.fit_predict(X)

# Determine optimal number of clusters (Elbow method)
inertia = []
for k in range(1, 11):
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(X)
    inertia.append(kmeans.inertia_)

plt.plot(range(1, 11), inertia, marker='o')
plt.xlabel('Number of clusters')
plt.ylabel('Inertia')
plt.title('Elbow Method')
plt.show()

# DBSCAN clustering
dbscan = DBSCAN(eps=0.5, min_samples=5)
clusters = dbscan.fit_predict(X)

# Hierarchical clustering
agg_clustering = AgglomerativeClustering(n_clusters=3)
clusters = agg_clustering.fit_predict(X)

# Gaussian Mixture Models
gmm = GaussianMixture(n_components=3, random_state=42)
clusters = gmm.fit_predict(X)
```

### Dimensionality Reduction
```python
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.manifold import TSNE

# PCA
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X)
print(f"Explained variance ratio: {pca.explained_variance_ratio_}")

# Plot PCA results
plt.figure(figsize=(10, 8))
plt.scatter(X_pca[:, 0], X_pca[:, 1], c=y, cmap='viridis')
plt.xlabel('First Principal Component')
plt.ylabel('Second Principal Component')
plt.title('PCA Results')
plt.colorbar(label='Class')
plt.show()

# t-SNE
tsne = TSNE(n_components=2, random_state=42)
X_tsne = tsne.fit_transform(X)

# Plot t-SNE results
plt.figure(figsize=(10, 8))
plt.scatter(X_tsne[:, 0], X_tsne[:, 1], c=y, cmap='viridis')
plt.xlabel('t-SNE Feature 1')
plt.ylabel('t-SNE Feature 2')
plt.title('t-SNE Results')
plt.colorbar(label='Class')
plt.show()
```


## References
We refer to the official documentation of scikit-learn: [https://scikit-learn.org/stable/](https://scikit-learn.org/stable/)
