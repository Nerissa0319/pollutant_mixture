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
import multiprocessing
import pickle
from create_dirs import *

# dengue_df = pd.read_csv(f'{data_dir}/dengue_lag.csv', header=0, index_col=0)
# df_feature, imp_feature = lasso_feature_selection(dengue_df,'Incidence_Rate')
# selected_features = list(df_feature['feature'])
# variables = selected_features.copy()

# num_covariates = len(variables)
# X = dengue_df[variables]
# X = X.values
# X = X.astype(np.float64)


def compute_IRR(args):
    disease,lag_num = args
    data1 = pd.read_csv(f'{data_dir}/{disease}_lag.csv', header=0, index_col=0)
    data = data1.copy()
    for c in data.columns:
        if 'Incidence_Rate' in c:
            data[c] = data1[c] * 100000
    air_pollutants = ['pm25','pm10','co','so2','no2','o3']
    predictor_lag = []
    weather = ['MeanT','AH','Rain']
    variables = []
    predictor_lag.extend(air_pollutants)
    variables.extend(air_pollutants)
    variables.extend(weather)
    if lag_num!=0:
        for i in range(lag_num):
            i = i + 1
            variables.append(f'Incidence_Rate_lag{i}')
            predictor_lag.append(f'Incidence_Rate_lag{i}')
        for i in range(lag_num):
            i = i + 1
            for ap in air_pollutants:
                variables.append(f'{ap}_lag{i}')
                predictor_lag.append(f'{ap}_lag{i}')
            for w in weather:
                variables.append(f'{w}_lag{i}')
    X = data[variables]
    num_covariates = len(variables)
    X = X.values
    X = X.astype(np.float64)

    def irr(feature_ind,x2_ind):
        print(disease,'\n')
    
        print(f'{variables[feature_ind]}, {variables[x2_ind]}\n')
        
        saveto = f'{gpplot_dir}/{disease}/lag{lag_num}/Incidence_Rate'

        if os.path.exists(f'{saveto}/model_rate.csv'):  
            print(f'{disease},lag{lag_num}\n')
            if os.path.exists(
                f'{saveto}/pdp/{variables[feature_ind]}/irr_{variables[feature_ind]}_{variables[x2_ind]}.csv'
                ):
                print('plot exists!')
                return
            else:
                if not os.path.exists(f'{saveto}/pdp/{variables[feature_ind]}'):
                    os.makedirs(f'{saveto}/pdp/{variables[feature_ind]}')
                with open(f'{saveto}/model_rate','rb') as f:
                    model_rate = pickle.load(f)
                with open(f'{saveto}/bs_rate','rb') as f:
                    bootstrap_X = pickle.load(f)
                print('load\n')
                feature_val = np.linspace(np.min(X[:, feature_ind]), np.max(X[:, feature_ind]), 100)
                feature_df = np.median(X, axis=0)
                fixed_features = np.tile(feature_df, (len(feature_val), 1))
                mean_rate, var_rate = model_rate.predict_y(fixed_features)
                fixed_features[:, feature_ind] = feature_val
                pred_mean_rate, pred_var_rate = model_rate.predict_y(fixed_features)
                # incidence_rate_ratio
                incidence_rate_ratio = pred_mean_rate / mean_rate

                x2_change1 = np.percentile(X[x2_ind], 5, axis=0)
                x2_change2 = np.percentile(X[x2_ind], 95, axis=0)
                features_change1 = feature_df.copy()
                features_change1[x2_ind] = x2_change1
                features_change2 = feature_df.copy()
                features_change2[x2_ind] = x2_change2
                fixed_features_change1 = np.tile(features_change1, (len(feature_val), 1))
                fixed_features_change2 = np.tile(features_change2, (len(feature_val), 1))
                fixed_features_change1[:, feature_ind] = feature_val
                fixed_features_change2[:, feature_ind] = feature_val
                pred_change1_rate, var_change1_rate = model_rate.predict_y(fixed_features_change1)
                pred_change2_rate, var_change2_rate = model_rate.predict_y(fixed_features_change2)

                irr_change1 = pred_change1_rate / mean_rate
                irr_change2 = pred_change2_rate / mean_rate
                irr_bootstrap_samples = []
                irr_bootstrap_samples1 = []
                irr_bootstrap_samples2 = []
                count = 0
                for s in bootstrap_X:
                    count += 1
                    if (count == 1) or (count ==1000):
                        print(count)
                    # Resample with replacement
                    # feature_val_sample = np.linspace(np.min(s[:, feature_ind]), np.max(s[:, feature_ind]), 100)
                    feature_df_sample = np.median(s, axis=0)
                    fixed_features_sample = np.tile(feature_df_sample, (len(feature_val), 1))
                    mean_rate_sample, var_rate_sample = model_rate.predict_y(fixed_features_sample)
                    fixed_features_sample[:, feature_ind] = feature_val
                    pred_mean_rate_sample, pred_var_rate_sample = model_rate.predict_y(fixed_features_sample)
                    feature_df_sample1 = np.median(s, axis=0)
                    feature_df_sample2 = np.median(s, axis=0)
                    x2_sample_change1 = np.percentile(s[x2_ind], 5, axis=0)
                    x2_sample_change2 = np.percentile(s[x2_ind], 95, axis=0)
                    feature_df_sample1[x2_ind] = x2_sample_change1
                    feature_df_sample2[x2_ind] = x2_sample_change2
                    fixed_features_sample1 = np.tile(feature_df_sample1, (len(feature_val), 1))
                    fixed_features_sample2 = np.tile(feature_df_sample2, (len(feature_val), 1))
                    fixed_features_sample1[:, feature_ind] = feature_val
                    fixed_features_sample2[:, feature_ind] = feature_val
                    pred_mean_sample1, pred_var_sample1 = model_rate.predict_y(fixed_features_sample1)
                    pred_mean_sample2, pred_var_sample2 = model_rate.predict_y(fixed_features_sample2)
                    # incidence_rate_ratio
                    incidence_rate_ratio_sample = pred_mean_rate_sample / mean_rate_sample
                    irr_bootstrap_samples.append(incidence_rate_ratio_sample)
                    incidence_rate_ratio_sample1 = pred_mean_sample1 / mean_rate_sample
                    incidence_rate_ratio_sample2 = pred_mean_sample2 / mean_rate_sample
                
                    irr_bootstrap_samples1.append(incidence_rate_ratio_sample1)
                    irr_bootstrap_samples2.append(incidence_rate_ratio_sample2)
                
                irr_bootstrap_samples = np.array(irr_bootstrap_samples)
                irr_ci_lower = np.percentile(irr_bootstrap_samples, 2.5, axis=0)
                irr_ci_upper = np.percentile(irr_bootstrap_samples, 97.5, axis=0)
                

                
                plt.plot(feature_val, incidence_rate_ratio, label='50% percentile',color='blue')
                plt.fill_between(feature_val, irr_ci_lower.squeeze(), irr_ci_upper.squeeze(), alpha=0.2,color='blue')
                plt.axhline(y=1, linestyle=':')
                plt.plot(feature_val, irr_change1, label=f'5% percentile',
                        color='red')
                plt.plot(feature_val, irr_change2, label=f'95% percentile',
                        color='orange')
                irr_bootstrap_samples1 = np.array(irr_bootstrap_samples1)
                irr_bootstrap_samples2 = np.array(irr_bootstrap_samples2)
                irr_ci_lower1 = np.percentile(irr_bootstrap_samples1, 2.5, axis=0)
                irr_ci_upper1 = np.percentile(irr_bootstrap_samples1, 97.5, axis=0)
                irr_ci_lower2 = np.percentile(irr_bootstrap_samples2, 2.5, axis=0)
                irr_ci_upper2 = np.percentile(irr_bootstrap_samples2, 97.5, axis=0)
                plt.fill_between(feature_val, irr_ci_upper1.squeeze(), irr_ci_lower1.squeeze(),
                                alpha=0.2,color='red')
                plt.fill_between(feature_val, irr_ci_upper2.squeeze(), irr_ci_lower2.squeeze(),
                                alpha=0.2,color='orange')
                gp_pdp = {
                    'feature_val': feature_val,
                    '50pctl_irr': incidence_rate_ratio,
                    '50pctl_ci_lower': irr_ci_lower.squeeze(),
                    '50pctl_ci_upper': irr_ci_upper.squeeze(),
                    '5pctl_irr':irr_change1,
                    '95pctl_irr':irr_change2,
                    '5pctl_ci_lower':irr_ci_lower1.squeeze(),
                    '5pctl_ci_upper':irr_ci_upper1.squeeze(),
                    '95pctl_ci_lower':irr_ci_lower2.squeeze(),
                    '95ptcl_ci_upper':irr_ci_upper2.squeeze()

                }
                gp_pdp = pd.DataFrame(gp_pdp)
                gp_pdp.to_csv(f'{saveto}/pdp/{variables[feature_ind]}/irr_{variables[feature_ind]}_{variables[x2_ind]}.csv')
                plt.xlabel(f'{variables[feature_ind]}')
                plt.ylabel('Incidence Rate Ratio')
                plt.title(f'{variables[feature_ind]} with {variables[x2_ind]}')
                plt.legend()
                plt.tight_layout()
                
                plt.savefig(
                    f'{saveto}/pdp/{variables[feature_ind]}/Incidence Rate Ratio for {variables[feature_ind]} with Effect of {variables[x2_ind]} change')

                plt.close()
        else:
            print('model not exists')
    for feature_index in range(num_covariates):
        for x2_index in range(num_covariates):
            x1 = variables[feature_index]
            x2 = variables[x2_index]
            flag = False
            if x1 in predictor_lag: 
                if x2 in predictor_lag:
                    if x1 != x2:
                        if 'lag' in x1:
                            if 'lag' in x2:
                                if x1[-1] == x2[-1]:
                                    flag = True
                                    print('condition1')
            if x1 in predictor_lag:
                if x2 in predictor_lag:
                    if x1 != x2:
                        if 'lag' not in x1:
                            if 'lag' not in x2:
                                print('condition2')
                                flag = True
            if flag:
                irr(feature_index,x2_index)
                    

def compute_total_dependence(disease,lag_num,X, num_data):
    saveto = f'{gpplot_dir}/{disease}/lag{lag_num}/Incidence_Rate'
    if not os.path.exists(f'{saveto}/model.csv'): 
        print('model not exists')
    else:
        with open(f'{saveto}/model_rate','rb') as f:
            model_rate = pickle.load(f)
        with open(f'{saveto}/bs_rate','rb') as f:
            bootstrap_X = pickle.load(f)

        
        feature_df = np.percentile(X, 50, axis=0)
        fixed_features = np.tile(feature_df, (num_data, 1))
        mean_rate, var_rate = model_rate.predict_y(fixed_features)
        for feature_ind in range(num_covariates):
            percentiles = np.arange(1,101)
            feature_val = np.percentile(X[:, feature_ind],percentiles)
            fixed_features[:, feature_ind] = feature_val
        
        pred_mean, pred_var = model_rate.predict_y(fixed_features)
        irr = pred_mean/mean_rate
        irr_bootstrap_samples = []
        count = 0
        for s in bootstrap_X:
            count += 1
            if count == 1000:
                print(count)
            # Resample with replacement
            feature_df_sample = np.percentile(s, 50, axis=0)
            fixed_features_sample = np.tile(feature_df_sample, (num_data, 1))
            mean_rate_sample, var_rate_sample = model_rate.predict_y(fixed_features_sample)
            for feature_ind in range(num_covariates):
                percentiles = np.arange(1,101)
                feature_val_sample = np.percentile(s[:, feature_ind],percentiles)
                fixed_features_sample[:, feature_ind] = feature_val_sample
            pred_mean_rate_sample, pred_var_rate_sample = model_rate.predict_y(fixed_features_sample)
            # incidence_rate_ratio
            incidence_rate_ratio_sample = pred_mean_rate_sample / mean_rate_sample
            irr_bootstrap_samples.append(incidence_rate_ratio_sample)
        
        
        irr_bootstrap_samples = np.array(irr_bootstrap_samples)
        irr_ci_lower = np.percentile(irr_bootstrap_samples, 2.5, axis=0)
        irr_ci_upper = np.percentile(irr_bootstrap_samples, 97.5, axis=0)
        plt_x = []
        for i in range(num_data):
            plt_x.append(i + 1)
        df_total = {
            'plt_x':plt_x,
            'ci_lower':irr_ci_lower.squeeze(),
            'ci_upper':irr_ci_upper.squeeze(),
            'irr':irr
        }
        df_total = pd.DataFrame(df_total)
        df_total.to_csv(f'{saveto}/total_dependence.csv')
        plt.plot(plt_x, irr)
        plt.axhline(y=1, linestyle=':')
        plt.fill_between(plt_x, irr_ci_lower.squeeze(), irr_ci_upper.squeeze(), color='gray', alpha=0.1)
        plt.xlabel(f'%')
        plt.ylabel('Incidence Rate Ratio (Reference at 50 Percentile)')
        plt.legend()
        plt.title(f'Total Dependence')
        plt.tight_layout()
        plt.savefig(f'{saveto}/Total Dependence')
        plt.close()





if __name__ == '__main__':
    diseases = ['Dengue Fever','HFMD','Campylobacter enteritis',
    'Salmonellosis(non-enteric fevers)','Acute Upper Respiratory Tract infections',
    'Acute Conjunctivitis', 'Acute Diarrhoea']    

    diseases1 =[]

    param_list = [(disease, lag_num) for disease in diseases for lag_num in range(9)]
    # param_list = [(X, feature_index, x2_index) for (feature_index, x2_index) in it.product(list(range(10,23)),list(range(num_covariates)))]
    pool = multiprocessing.Pool(processes =20)
                
    # Use multiprocessing to parallelize the execution of incidence_model
    pool.map(compute_IRR, param_list)

    # Close the pool to free up resources
    pool.close()
    pool.join()

    # for disease in diseases:
    #     for lag_num in range(2):
    #         param_list = (disease,lag_num)
    #         compute_IRR(param_list)
    for disease in diseases:
        for lag_num in range(9):
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
            if lag_num!=0:
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
            Y = data[disease]
            X = X.values
            X = X.astype(np.float64)
            compute_total_dependence(disease,lag_num,X, 100)