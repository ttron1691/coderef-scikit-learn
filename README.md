# Code Reference for scikit-learn
## Install Packages
```Bash
pip install -U scikit-learn
```
## Linear models
```Python
from sklearn import linear_model
reg = linear_model.LinearRegression()
reg.fit([[0, 0], [1, 1], [2, 2]], [0, 1, 2])
reg.coef_
```
