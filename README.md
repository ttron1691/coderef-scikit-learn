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

## References
We refer to the official documentation of scikit-learn: [https://scikit-learn.org/stable/](https://scikit-learn.org/stable/)
