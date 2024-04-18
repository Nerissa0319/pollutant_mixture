from sklearn.linear_model import Lasso
import os
import pandas as pd
from sklearn.linear_model import LassoCV
from sklearn.model_selection import train_test_split
from sklearn.linear_model import ElasticNet
from sklearn.linear_model import ElasticNetCV
import numpy as np
# import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler,MinMaxScaler
def lasso_feature_selection(dataframe,yname,disease):

    data = dataframe.copy()
    if yname == 'Incidence_Rate':
        data[yname] = data[yname] * 100000
    for col in data.columns:
        scaler = StandardScaler()
        data[col] = scaler.fit_transform(data[col].values.reshape(-1,1))
    if yname == 'Incidence_Rate':
        X = data.iloc[:, list(range(1, 10)) + list(range(21,29)) + list(range(29, 101))]
        y = data[yname]
    else:
        X = data.iloc[:, list(range(1, 10)) + list(range(13, 21)) + list(range(29, 101))]
        y = data[disease]
    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    # Instantiate LassoCV with a range of alpha values
    # Define a range of alpha values on a logarithmic scale
    alpha_min = 1e-6
    alpha_max = 10.0
    num_alphas = 20

    lambdas = np.logspace(np.log10(alpha_min), np.log10(alpha_max), num=num_alphas)
    # lambdas = [0.00001,0.0001,0.001, 0.01, 0.1, 1.0]
    lasso_coefficients = []
    for l in lambdas:
        lasso = Lasso(alpha=l, max_iter=10000)
        lasso.fit(X_train, y_train)
        lasso_coefficients.append(lasso.coef_)
    # plt.plot(lambdas, lasso_coefficients)
    # plt.xscale('log')
    # plt.xlabel('Lambda')
    # plt.ylabel('Coefficients')
    # plt.tight_layout()
    # plt.show()
    # plt.close()
    lasso_cv = LassoCV(alphas=lambdas, cv=5, max_iter=10000)
    lasso_cv.fit(X_train, y_train)
    lasso_best_alpha = lasso_cv.alpha_
    lasso_model = Lasso(alpha=lasso_best_alpha, max_iter=10000)
    lasso_model.fit(X_train, y_train)
    if yname == 'Incidence_Rate':
        dic = {'feature': list(data.columns)[1:10] + list(data.columns)[21:29] + list(data.columns)[29:101],
           'coeff': lasso_model.coef_}
    else:
        dic = {'feature': list(data.columns)[1:10] + list(data.columns)[13:21] + list(data.columns)[29:101],
           'coeff': lasso_model.coef_}
    df = pd.DataFrame(dic)
    df1 = df[df['coeff'] != 0]
    coef = pd.Series(lasso_model.coef_,
                     index=list(data.columns)[1:10] + list(data.columns)[13:21] + list(data.columns)[29:101])
    imp_coef = pd.concat([coef.sort_values().head(10), coef.sort_values().tail(10)])
    # sns.set(font_scale=1.2)
    # imp_coef.plot(kind='barh')
    # plt.title('Lasso')
    # plt.tight_layout()
    # plt.show()
    return df1, imp_coef

