import gpflow
import os
import pandas as pd
import numpy as np
import matplotlib
import matplotlib.dates as md
from datetime import datetime,date
from sklearn.model_selection import train_test_split
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import tensorflow as tf
import itertools as it
from sklearn.model_selection import cross_val_score
from gpflow.expectations import expectation
from gpflow.mean_functions import Zero
from sklearn.utils import resample
from gpflow.kernels import RBF, Matern12, Matern32, Matern52
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import make_scorer, mean_squared_error
from feature_selection import lasso_feature_selection
import pickle
from bbs import *
from gp_abic import *
import multiprocessing
# os.chdir(f'{os.getcwd()}//src')

from create_dirs import *

def incidence_rate_model(disease,lag_num):

    print(disease,'\n')
    saveto = f'{gpplot_dir}/{disease}/lag{lag_num}/Incidence_Rate'
    data1 = pd.read_csv(f'{data_dir}/{disease}_lag.csv', header=0, index_col=0)
    data = data1.copy()
    for c in data.columns:
        if 'Incidence_Rate' in c:
            data[c] = data1[c] * 100000
    air_pollutants = ['pm25','pm10','co','so2','no2','o3']
    weather = ['MeanT','AH','Rain']
    variables = []

    variables.extend(air_pollutants)
    variables.extend(weather)
    if lag_num != 0:
        for i in range(lag_num):
            i = i + 1
            variables.append(f'Incidence_Rate_lag{i}')
        for i in range(lag_num):
            i = i + 1
            for ap in air_pollutants:
                variables.append(f'{ap}_lag{i}')
            for w in weather:
                variables.append(f'{w}_lag{i}')

    X = data[variables]
    num_covariates = len(variables)
    Y = data['Incidence_Rate']
    X = X.values
    X = X.astype(np.float64)
    Y = Y.values.reshape(len(Y), 1)
    Y = Y.astype(np.float64)
    if not os.path.exists(f'{saveto}/model_rate'):
        
        if not os.path.exists(saveto):
            os.makedirs(saveto)
        

        kernels  = [gpflow.kernels.RBF(), gpflow.kernels.Matern12(),
                gpflow.kernels.Matern32(),
                gpflow.kernels.Exponential()]
      
        min_rmse = np.inf
        best_kernel = None
        for kernel in kernels:
            rmse = 0.0
            for fold in range(5):
                print(f'{disease} lag{lag_num} - fold:{fold}\n')
                start = len(X) * fold // 5
                end = len(X) * (fold + 1) // 5
                X_train = np.concatenate([X[:start], X[end:]])
                Y_train = np.concatenate([Y[:start], Y[end:]])
                X_test = X[start:end]
                Y_test = Y[start:end]

                model = gpflow.models.VGP(data=(X_train, Y_train), kernel=kernel, likelihood=gpflow.likelihoods.Poisson())
                gpflow.optimizers.Scipy().minimize(model.training_loss, model.trainable_variables)
                print('model done')
                Y_pred, _ = model.predict_y(X_test)
                rmse += np.sqrt(np.mean((Y_pred - Y_test) ** 2))
                print('predict done')
            mean_rmse = rmse / 5
            if mean_rmse < min_rmse:
                min_rmse = mean_rmse
                best_kernel = kernel
            print(f"{disease} lag{lag_num} - Kernel: {kernel}, Mean RMSE: {mean_rmse}\n")

        print(f'Best Kernels is : {best_kernel} and the RMSE is {min_rmse}')
        f = open(f'{saveto}/infor.txt','w')
        f.write(f'Incidence Model for {disease} lag {lag_num}\n')
        f.write(f'Best Kernels for {disease} lag{lag_num} is : {best_kernel} and the RMSE is {min_rmse}\n')
        
        
        # Define the kernel for the Gaussian process
        kernel = best_kernel
        model_rate = gpflow.models.VGP(data=(X, Y), kernel=kernel, likelihood=gpflow.likelihoods.Poisson())
        opt_rate = gpflow.optimizers.Scipy()
        objective_rate = model_rate.training_loss
        
        opt_logs_rate = opt_rate.minimize(objective_rate, variables=model_rate.trainable_variables)
        # Make predictions with the trained model
        # split_index = int(0.8 * len(X))
        # X_train = X[:split_index]
        # X_test = X[split_index:]
        # Y_train = Y[:split_index]
        # Y_test = Y[split_index:]
        
        bootstrap_samples = block_bootstrap(X, 20, 1000)
        with open(f'{saveto}/model_rate','wb') as m:
            pickle.dump(model_rate,m)
        with open(f'{saveto}/bs_rate','wb') as fp:
            pickle.dump(bootstrap_samples,fp)

        aic = gp_aic(model_rate,Y,X)
        print('AIC: ',aic)
        bic = gp_bic(model_rate,Y,X)
        print('BIC: ',bic)
        f.write(f'AIC: {aic.numpy()}\n')
        f.write(f'BIC: {bic.numpy()}\n')
        f.close()


        pred_mean, pred_var = model_rate.predict_y(X)
        datex = []
        for ind in data.index:
            datex.append(md.date2num(datetime.strptime(ind,'%Y-%m-%d')))
        fig,ax = plt.subplots()
        ax.xaxis.set_major_locator(md.YearLocator())
        ax.xaxis.set_major_formatter(md.DateFormatter('%Y'))
        # mean1 = y_scaler.inverse_transform(mean)
        ratio = pred_mean[:, 0] / Y[:, 0]
        plt.plot(datex, ratio)
        plt.savefig(f'{saveto}/ratio')
        plt.close()
        fig,ax = plt.subplots()
        ax.xaxis.set_major_locator(md.YearLocator())
        ax.xaxis.set_major_formatter(md.DateFormatter('%Y'))
        plt.plot(datex, pred_mean[:, 0], color='green', label='Predictions', lw=1)
        plt.scatter(datex, Y[:, 0], color='red', label='Observations', s=3)
        std = np.sqrt(pred_var)
        upper = pred_mean + 1.96 * std
        lower = pred_mean - 1.96 * std
        plt.fill_between(datex, tf.squeeze(upper), tf.squeeze(lower),
                        alpha=0.5)
        plt.title('Predicted Incidence Rate')
        plt.legend()
        plt.tight_layout()
        plt.savefig(f'{saveto}/Predicted Values VS Observed Values')
        plt.close()
        model_predict = {
            'datex':datex,
            'pred_to_obs':ratio,
            'pred':pred_mean[:,0],
            'std':std[:,0],
            'ci_upper':upper[:,0],
            'ci_lower':lower[:,0]
        }
        model_predict = pd.DataFrame(model_predict)
        model_predict.to_csv(f'{saveto}/Predicted Values VS Observed Values.csv')
    else:
        print('model exists')
        with open(f'{saveto}/model_rate','rb') as f:
            model_rate = pickle.load(f)
        f1 = open(f'{saveto}/infor.txt','a')
        aic = gp_aic(model_rate,Y,X)
        print('AIC: ',aic)
        bic = gp_bic(model_rate,Y,X)
        print('BIC: ',bic)
        f1.write(f'AIC: {aic.numpy()}\n')
        f1.write(f'BIC: {bic.numpy()}\n')
        f1.close()
        if not os.path.exists(f'{saveto}/Predicted Values VS Observed Values.csv'):
            print('generating csv')
            pred_mean, pred_var = model_rate.predict_y(X)
            datex = []
            for ind in data.index:
                datex.append(md.date2num(datetime.strptime(ind,'%Y-%m-%d')))
            fig,ax = plt.subplots()
            ax.xaxis.set_major_locator(md.YearLocator())
            ax.xaxis.set_major_formatter(md.DateFormatter('%Y'))
            # mean1 = y_scaler.inverse_transform(mean)
            ratio = pred_mean[:, 0] / Y[:, 0]
            plt.plot(datex, ratio)
            plt.savefig(f'{saveto}/ratio')
            plt.close()
            fig,ax = plt.subplots()
            ax.xaxis.set_major_locator(md.YearLocator())
            ax.xaxis.set_major_formatter(md.DateFormatter('%Y'))
            plt.plot(datex, pred_mean[:, 0], color='green', label='Predictions', lw=1)
            plt.scatter(datex, Y[:, 0], color='red', label='Observations', s=3)
            std = np.sqrt(pred_var)
            upper = pred_mean + 1.96 * std
            lower = pred_mean - 1.96 * std
            plt.fill_between(datex, tf.squeeze(upper), tf.squeeze(lower),
                            alpha=0.5)
            plt.title('Predicted Incidence Rate')
            plt.legend()
            plt.tight_layout()
            plt.savefig(f'{saveto}/Predicted Values VS Observed Values')
            plt.close()
            model_predict = {
                'datex':datex,
                'pred_to_obs':ratio,
                'pred':pred_mean[:,0],
                'std':std[:,0],
                'ci_upper':upper[:,0],
                'ci_lower':lower[:,0]
            }
        
            model_predict = pd.DataFrame(model_predict)
            model_predict.to_csv(f'{saveto}/Predicted Values VS Observed Values.csv')
        else:
            print('model and data exist')
        

diseases = ['Dengue Fever','HFMD','Campylobacter enteritis',
'Salmonellosis(non-enteric fevers)','Acute Upper Respiratory Tract infections',
'Acute Conjunctivitis', 'Acute Diarrhoea']    

lags_range = range(9)      

# for disease in diseases:
#     for lag in lags_range:

#         incidence_rate_model(disease,lag)

pool = multiprocessing.Pool(processes=50)  # Use the number of CPU cores

# Use multiprocessing to parallelize the execution of incidence_model
pool.starmap(incidence_rate_model, [(disease, lag) for disease in diseases for lag in lags_range])

# Close the pool to free up resources
pool.close()
pool.join()