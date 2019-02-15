import pandas as pd
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.metrics import mean_squared_error
from math import sqrt

"""
    1. CRIM      per capita crime rate by town
    2. ZN        proportion of residential land zoned for lots over 
                 25,000 sq.ft.
    3. INDUS     proportion of non-retail business acres per town
    4. CHAS      Charles River dummy variable (= 1 if tract bounds 
                 river; 0 otherwise)
    5. NOX       nitric oxides concentration (parts per 10 million)
    6. RM        average number of rooms per dwelling
    7. AGE       proportion of owner-occupied units built prior to 1940
    8. DIS       weighted distances to five Boston employment centres
    9. RAD       index of accessibility to radial highways
    10. TAX      full-value property-tax rate per $10,000
    11. PTRATIO  pupil-teacher ratio by town
    12. B        1000(Bk - 0.63)^2 where Bk is the proportion of blacks 
                 by town
    13. LSTAT    % lower status of the population
    14. MEDV     Median value of owner-occupied homes in $1000's
"""
    
columns = ['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 'DIS', 'RAD', 'TAX', 'PTRATIO', 'B', 'LSTATE', 'MEDV']

dataset = pd.read_csv('housing.data.txt', names=columns, sep=r'\s{1,}')

unscale_max = dataset.max()
unscale_min = dataset.min()
# norm scale dataset
dataset -= dataset.min()
dataset /= dataset.max()

parameters = {
    'criterion': ['mse'],
    'max_depth': [1, 3, 5, 7, 9, 11],
    'max_features': [5, 7, 9, 11, 13],
    'min_samples_leaf': [1, 3, 5, 10, 20],
    'random_state': [42]
}

clf = GridSearchCV(DecisionTreeRegressor(), parameters, cv=5)
X = dataset.loc[:, dataset.columns != 'MEDV']
Y = dataset.loc[:, dataset.columns == 'MEDV']

clf.fit(X, Y)
print(clf.best_params_)

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.33, random_state=42)

dt = DecisionTreeRegressor(criterion='mse', max_depth=11, max_features=7, min_samples_leaf=5, random_state=42)

dt.fit(X_train, y_train)

y_pred = dt.predict(X_test)

print('RMSE', sqrt(mean_squared_error(y_test, y_pred)))
print('MSE', mean_squared_error(y_test, y_pred))

unscaled_y_pred = (y_pred * unscale_max['MEDV']) + unscale_min['MEDV']
unscaled_y_test =  (y_test * unscale_max['MEDV']) + unscale_min['MEDV']

unscaled_y_pred = pd.DataFrame(unscaled_y_pred, index=y_test.index, columns=['MEDV'])
# print(unscaled_y_pred, unscaled_y_test)