import pandas as pd
import numpy as np
import statsmodels.api as sm
import matplotlib.dates as md
from datetime import datetime,date
import matplotlib.pyplot as plt
import os
import scipy.io
import seaborn as sns
import multiprocessing
from statsmodels.tsa.arima.model import ARIMA
from sklearn.preprocessing import StandardScaler
from feature_selection import lasso_feature_selection
import pickle
from create_dirs import *
from gp_abic import *

def glm(yname,disease):
    saveto = f'{plot_dir}/GLM/{disease}/{yname}'
    if not os.path.exists(saveto):
        os.makedirs(saveto)
    disease_df1 = pd.read_csv(f'{data_dir}/{disease}_lag.csv',header=0,index_col=0)
    df_feature, imp_feature = lasso_feature_selection(disease_df1,yname,disease)
    selected_features = list(df_feature['feature'])
    independent_vars = selected_features


    # print('---------------------------Poisson--------')
    df = disease_df1.copy()
    # for col in df.columns:
    #     scaler = StandardScaler()
    #     df[col] = scaler.fit_transform(df[col].values.reshape(-1,1))
    for c in df.columns:
        if 'Incidence_Rate' in c:
            df[c] = disease_df1[c] * 100000
    
    X = df[independent_vars]
    if yname == 'Incidence_Rate':
        y = df[yname]

    else:
        y = df[disease]
    
    X = sm.add_constant(X)
    final_model = sm.GLM(y,X,family=sm.families.Poisson())
    results = final_model.fit(method='lbfgs')
    print(results.summary())
    file = open(f'{saveto}/info.txt','a')
    file.write(results.summary().as_text())
    file.write('\n')
    pred_frame = results.get_prediction(X).summary_frame()
    y_pred = pred_frame['mean']
    y_pred_se = pred_frame['mean_se']
    y_lower = pred_frame['mean_ci_lower']
    y_upper = pred_frame['mean_ci_upper']
    datex = []
    for ind in disease_df1.index:
        datex.append(md.date2num(datetime.strptime(ind,'%Y-%m-%d')))
    pred_data = {
        'datex': datex,
        'y_pred': y_pred,
        'y_obs': y,
        'ci_lower':y_lower,
        'ci_upper':y_upper,
        'y_se':y_pred_se
    }
    
    pred_df = pd.DataFrame(pred_data)
    pred_df = pd.concat([pred_df,X],axis = 1)
    pred_df.to_csv(f'{saveto}/pred_df.csv')
    # fig,ax = plt.subplots()
    # ax.xaxis.set_major_locator(md.YearLocator())
    # ax.xaxis.set_major_formatter(md.DateFormatter('%Y'))
    # plt.plot(datex,y_pred,label='Predicted Values',lw=1)
    # plt.scatter(datex,y,label='Observed Values',color='red',s=3)
    # plt.title('GLM')
    # plt.savefig(f'{saveto}/GLM_{disease}_{yname}')
    # plt.close()
    with open(f'{saveto}/glm_{disease}_{yname}','wb') as f:
        pickle.dump(results,f)
    print(f'AIC for GLM_{yname}: {results.aic}')
    print(f'BIC for GLM_{yname}: {results.bic_llf}')
    file.write(f'AIC for GLM_{yname}: {results.aic}\n')
    file.write(f'BIC for GLM_{yname}: {results.bic_llf}\n')
    file.close()

def glmlag(yname,disease,lagnum):
    saveto = f'{plot_dir}/GLM/{disease}/{yname}/lag{lagnum}'
    if not os.path.exists(saveto):
        os.makedirs(saveto)
    disease_df1 = pd.read_csv(f'{data_dir}/{disease}_lag.csv',header=0,index_col=0)
    air_pollutants = ['pm25','pm10','co','so2','no2','o3']
    weather = ['MeanT','AH','Rain']
    variables = []
    if lagnum == 0:
        variables.extend(air_pollutants)
        variables.extend(weather)
    else:
        for i in range(lagnum):
            i = i + 1
            if yname == 'Incidence':
                variables.append(f'{disease}_lag{i}')
            else:
                variables.append(f'Incidence_Rate_lag{i}')
        for i in range(lagnum):
            i = i + 1
            for ap in air_pollutants:
                variables.append(f'{ap}_lag{i}')
            for w in weather:
                variables.append(f'{w}_lag{i}')

    # print('---------------------------Poisson--------')
    df = disease_df1.copy()
    # for col in df.columns:
    #     scaler = StandardScaler()
    #     df[col] = scaler.fit_transform(df[col].values.reshape(-1,1))
    for c in df.columns:
        if 'Incidence_Rate' in c:
            df[c] = disease_df1[c] * 100000
    
    
    X = df[variables]
    num_covariates = len(variables)


    # print('---------------------------Poisson--------')
    if yname == 'Incidence_Rate':
        y = df[yname]

    else:
        y = df[disease]

    X = sm.add_constant(X)
    final_model = sm.GLM(y,X,family=sm.families.Poisson())
    results = final_model.fit(method='lbfgs')
    file = open(f'{saveto}/info.txt','a')
    file.write(results.summary().as_text())
    file.write('\n')
    print(results.summary())
    pred_frame = results.get_prediction(X).summary_frame()
    y_pred = pred_frame['mean']
    y_pred_se = pred_frame['mean_se']
    y_lower = pred_frame['mean_ci_lower']
    y_upper = pred_frame['mean_ci_upper']
    datex = []
    for ind in disease_df1.index:
        datex.append(md.date2num(datetime.strptime(ind,'%Y-%m-%d')))
    pred_data = {
        'datex': datex,
        'y_pred': y_pred,
        'y_obs': y,
        'ci_lower':y_lower,
        'ci_upper':y_upper,
        'y_se':y_pred_se
    }
    pred_df = pd.DataFrame(pred_data)
    pred_df = pd.concat([pred_df,X],axis = 1)
    pred_df.to_csv(f'{saveto}/GLM_{disease}_{yname}_pred_df.csv')
    # fig,ax = plt.subplots()
    # ax.xaxis.set_major_locator(md.YearLocator())
    # ax.xaxis.set_major_formatter(md.DateFormatter('%Y'))
    # plt.plot(datex,y_pred,label='Predicted Values',lw=1)
    # plt.scatter(datex,y,label='Observed Values',color='red',s=3)
    # plt.title('GLM')
    # plt.savefig(f'{saveto}/GLM_{disease}_{yname}')
    # plt.close()
    with open(f'{saveto}/glm_{disease}_{yname}','wb') as f:
        pickle.dump(results,f)
    print(f'AIC for GLM_{yname}: {results.aic}')
    print(f'BIC for GLM_{yname}: {results.bic_llf}')
    file.write(f'AIC for GLM_{yname}: {results.aic}\n')
    file.write(f'BIC for GLM_{yname}: {results.bic_llf}\n')
    file.close()
if __name__ == '__main__':
    
    diseases = ['Dengue Fever','HFMD','Campylobacter enteritis',
    'Salmonellosis(non-enteric fevers)','Acute Upper Respiratory Tract infections',
    'Acute Conjunctivitis', 'Acute Diarrhoea']    
 
    ytype = ['Incidence','Incidence_Rate']
    for d in diseases:
        for t in ytype:
            glm(t,d)
    pool = multiprocessing.Pool(processes=10)  # Use the number of CPU cores

    # # # Use multiprocessing to parallelize the execution of incidence_model
    # # pool.starmap(glm, [(ytype,disease) for disease in diseases for ytype in ['Incidence','Incidence_Rate']])

    # # # Close the pool to free up resources
    # # pool.close()
    # # pool.join()
    pool.starmap(glmlag,[(t,disease,l) for disease in diseases for t in ytype for l in range(9)])
    pool.close()
    pool.join()