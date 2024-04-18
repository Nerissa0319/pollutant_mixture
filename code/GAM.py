from datetime import date
import numpy as np
from pygam import PoissonGAM, s
import os
from datetime import datetime
from sklearn.model_selection import RandomizedSearchCV
import matplotlib.dates as md
from matplotlib.dates import DateFormatter
from sklearn.metrics import accuracy_score
from sklearn.metrics import log_loss
import pandas as pd
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
import itertools as it
import pickle
from sklearn.model_selection import train_test_split
from feature_selection import lasso_feature_selection
import matplotlib.pyplot as plt
from create_dirs import *
from gp_abic import *
import multiprocessing
import sys
from contextlib import redirect_stdout
import math
import scipy.stats as stats

def cv_spline(X, y, num_covariates):
    nsplines = [10]
    min_rmse = np.inf
    best_n = 0

    for ns in nsplines:
        rmse = 0.0
        rmse2 = 0.0
        for fold in range(5):
            print(fold)
            start = len(X) * fold // 5
            end = len(X) * (fold + 1) // 5

            X_train = np.concatenate([X[:start], X[end:]])
            y_train = np.concatenate([y[:start], y[end:]])
    

            X_test = X[start:end]
            y_test = y[start:end]


            gam = PoissonGAM(n_splines=ns).gridsearch(X_train, y_train)

            y_pred = gam.predict(X_test)

            rmse += np.sqrt(np.mean(y_pred - y_test) ** 2)

        mean_rmse = rmse / 5

        if mean_rmse < min_rmse:
            min_rmse = mean_rmse
            best_n = ns

    return best_n

def compute_se(lower, upper):
    confidence_level = 0.95
    z = np.abs(stats.norm.ppf((1 - confidence_level) / 2))  # z-score
    return (upper - lower) / (2 * z)


def compute_CI(se, coef):  # delta method to calculate confidence interval
    pdep0 = list(coef)[0]
    riskci = []
    oddsratio_ci = []
    odds_ci = []
    confidence_level = 0.95
    z = np.abs(stats.norm.ppf((1 - confidence_level) / 2))  # z-score
    for i in range(len(se)):
        derivative = np.exp(coef[i] - pdep0)  # derivative of odds ratio, i.e. derivative of exp(coef) = exp(coef)
        delta = se[i] * derivative
        margin_of_error = z * delta
        lower_bound = np.exp(coef[i] - pdep0) - margin_of_error
        upper_bound = np.exp(coef[i] - pdep0) + margin_of_error
        oddsratio_ci.append([lower_bound, upper_bound])
        riskci.append([(lower_bound - 1), (upper_bound - 1)])
        d_odds = np.exp(coef[i])
        delta_odds = se[i] * d_odds
        margin_odds = z * delta_odds
        odds_ci.append([np.exp(coef[i]) - margin_odds, np.exp(coef[i]) + margin_odds])

    return odds_ci, riskci, oddsratio_ci

def poissongam(yname,disease):
    saveto = f'{plot_dir}/GAM/{disease}/{yname}'
    if not os.path.exists(saveto):
        os.makedirs(saveto)
    if not os.path.exists(f'{saveto}/pdp.xlsx'):
        disease_df1 = pd.read_csv(f'{data_dir}/{disease}_lag.csv',header=0,index_col=0)
        df_feature, imp_feature = lasso_feature_selection(disease_df1,yname,disease)
        selected_features = list(df_feature['feature'])
        independent_vars = selected_features
        num_covariates = len(independent_vars)
        df = disease_df1.copy()
        for c in df.columns:
            if 'Incidence_Rate' in c:
                df[c] = disease_df1[c] * 100000
        # for col in df.columns:
        #     scaler = StandardScaler()
        #     df[col] = scaler.fit_transform(df[col].values.reshape(-1,1))
        # print('---------------------------Poisson--------')
        if yname == 'Incidence_Rate':
            y = df[yname]
        else:
            y = df[disease]
        X = df[independent_vars]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        best_n = cv_spline(X,y,num_covariates)
        gam = PoissonGAM(n_splines = best_n).fit(X,y)
        lams=np.random.rand(20,num_covariates)
        lams=lams*6-3
        lams=np.exp(lams)
        print('grid search')
        gam.gridsearch(X, y, lam=lams)
        datex = []
        for ind in df.index:
            datex.append(md.date2num(datetime.strptime(ind,'%Y-%m-%d')))
        best_lam = gam.lam
        gam_model =PoissonGAM(n_splines=best_n,lam = best_lam).fit(X,y)
        file = open(f'{saveto}/info.txt','w')
        with redirect_stdout(file):
            # Generate the summary and print it to the file
            gam_model.summary() 
        file.write('\n')
        predictions = gam_model.predict(X)
        conf_int = gam_model.confidence_intervals(X)
        pred_data = {
            'datex':datex,
            'y_pred':predictions,
            'y_obs':y,
            'ci_lower':conf_int.iloc[:,0],
            'ci_upper':conf_int.iloc[:,1]
        }
        pred_df = pd.DataFrame(pred_data)
        pred_df = pd.concat([pred_df,X],axis = 1)
        pred_df.to_csv(f'{saveto}/GAM_{yname}_pred_df.csv')

        for i in range(len(gam_model.terms)-1):
            XX = gam_model.generate_X_grid(term=i)
            pdep,confi = gam_model.partial_dependence(term=i,width=0.95)
            pdep0 = list(pdep)[0]
            upper_lim = confi[:, 1]
            lower_lim = confi[:, 0]
            se = compute_se(lower_lim, upper_lim)
            odds_ci, risk_ci, oddsratio_ci = compute_CI(se, pdep)
            dict = {
                'coef': list(pdep),
                'coef_lower': list(lower_lim),
                'coef_upper': list(upper_lim),
                'odds': list(np.exp(pdep)),
                'odds_lower': list(np.array(odds_ci)[:, 0]),
                'odds_upper': list(np.array(odds_ci)[:, 1]),
                'Odds Ratio': list(np.exp(pdep - pdep0)),
                'OR lower': list(np.array(oddsratio_ci)[:, 0]),
                'OR upper': list(np.array(oddsratio_ci)[:, 1]),
                'risk_increase': list((np.exp(pdep - pdep0) - 1)),
                'risk_lower': np.array(risk_ci)[:, 0],
                'risk_upper': np.array(risk_ci)[:, 1]
            }
            xnamei = X.columns[i]
            dict[xnamei] = list(XX[:,i])
            pdp_df = pd.DataFrame(dict)
            for ind, row in pdp_df.iterrows():
                if row['odds_lower'] < 1 and row['odds_upper'] < 1:
                    pdp_df.loc[ind, 'ci_sig1'] = 0.5
                elif row['odds_lower'] > 1 and row['odds_upper'] > 1:
                    pdp_df.loc[ind, 'ci_sig1'] = 0.5
                else:
                    pdp_df.loc[ind, 'ci_sig1'] = 0.3
                if row['risk_lower'] < 0 and row['risk_upper'] < 0:
                    pdp_df.loc[ind, 'ci_sig'] = 0.5
                elif row['risk_lower'] > 0 and row['risk_upper'] > 0:
                    pdp_df.loc[ind, 'ci_sig'] = 0.5
                else:
                    pdp_df.loc[ind, 'ci_sig'] = 0.3
            if os.path.exists(f'{saveto}/pdp.xlsx'):
                with pd.ExcelWriter(f'{saveto}/pdp.xlsx', mode='a') as writer:
                    pdp_df.to_excel(writer, sheet_name=f'{xnamei}', header=True, index=True)

            else:
                with pd.ExcelWriter(f'{saveto}/pdp.xlsx', mode='w') as writer:
                    pdp_df.to_excel(writer, sheet_name=f'{xnamei}', header=True, index=True)
        
        aic,bic = gam_abic(gam_model,y,X)
        print('AIC: ',aic)
        print('BIC: ',bic)

        file.write(f'AIC: {aic}')
        file.write('\n')
        file.write(f'BIC: {bic}')
        file.write('\n')
        file.close()
        
        with open(f'{saveto}/gam__{disease}_{yname}','wb') as f:
            pickle.dump(gam_model,f)
        with open(f'{saveto}/gam__{disease}_{yname}','wb') as f:
            pickle.dump(gam_model,f)
        return gam_model
    else:
        print('model exists\n')
        return


def poissongamlag(yname,disease,lagnum):
    # yname='Incidence_Rate'
    # disease='Dengue Fever'
    # lagnum=1
    saveto = f'{plot_dir}/GAM/{disease}/{yname}/lag{lagnum}'
    if not os.path.exists(saveto):
        os.makedirs(saveto)
    if not os.path.exists(f'{saveto}/pdp.xlsx'):
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
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        best_n = cv_spline(X,y,num_covariates)
        gam = PoissonGAM(n_splines = best_n).fit(X,y)
        lams=np.random.rand(20,num_covariates)
        lams=lams*6-3
        lams=10**lams
        print('grid search')
        gam.gridsearch(X, y, lam=lams)
        datex = []
        for ind in df.index:
            datex.append(md.date2num(datetime.strptime(ind,'%Y-%m-%d')))
        best_lam = gam.lam
        gam_model =PoissonGAM(n_splines=best_n,lam = best_lam).fit(X,y)    
            
        file = open(f'{saveto}/info.txt','w')
        with redirect_stdout(file):
            # Generate the summary and print it to the file
            gam_model.summary() 
        file.write('\n')
        predictions = gam_model.predict(X)
        conf_int = gam_model.confidence_intervals(X)
        pred_data = {
            'datex':datex,
            'y_pred':predictions,
            'y_obs':y,
            'ci_lower':conf_int[:,0],
            'ci_upper':conf_int[:,1]
        }
        pred_df = pd.DataFrame(pred_data)
        pred_df = pd.concat([pred_df,X],axis = 1)
        pred_df.to_csv(f'{saveto}/GAM_{yname}_pred_df.csv')

        for i in range(len(gam_model.terms)-1):
            XX = gam_model.generate_X_grid(term=i)
            pdep,confi = gam_model.partial_dependence(term=i,width=0.95)
            pdep0 = list(pdep)[0]
            upper_lim = confi[:, 1]
            lower_lim = confi[:, 0]
            se = compute_se(lower_lim, upper_lim)
            odds_ci, risk_ci, oddsratio_ci = compute_CI(se, pdep)
            dict = {
                'coef': list(pdep),
                'coef_lower': list(lower_lim),
                'coef_upper': list(upper_lim),
                'odds': list(np.exp(pdep)),
                'odds_lower': list(np.array(odds_ci)[:, 0]),
                'odds_upper': list(np.array(odds_ci)[:, 1]),
                'Odds Ratio': list(np.exp(pdep - pdep0)),
                'OR lower': list(np.array(oddsratio_ci)[:, 0]),
                'OR upper': list(np.array(oddsratio_ci)[:, 1]),
                'risk_increase': list((np.exp(pdep - pdep0) - 1)),
                'risk_lower': np.array(risk_ci)[:, 0],
                'risk_upper': np.array(risk_ci)[:, 1]
            }
            xnamei = X.columns[i]
            dict[xnamei] = list(XX[:,i])
            pdp_df = pd.DataFrame(dict)
            for ind, row in pdp_df.iterrows():
                if row['odds_lower'] < 1 and row['odds_upper'] < 1:
                    pdp_df.loc[ind, 'ci_sig1'] = 0.5
                elif row['odds_lower'] > 1 and row['odds_upper'] > 1:
                    pdp_df.loc[ind, 'ci_sig1'] = 0.5
                else:
                    pdp_df.loc[ind, 'ci_sig1'] = 0.3
                if row['risk_lower'] < 0 and row['risk_upper'] < 0:
                    pdp_df.loc[ind, 'ci_sig'] = 0.5
                elif row['risk_lower'] > 0 and row['risk_upper'] > 0:
                    pdp_df.loc[ind, 'ci_sig'] = 0.5
                else:
                    pdp_df.loc[ind, 'ci_sig'] = 0.3
            if os.path.exists(f'{saveto}/pdp.xlsx'):
                with pd.ExcelWriter(f'{saveto}/pdp.xlsx', mode='a') as writer:
                    pdp_df.to_excel(writer, sheet_name=f'{xnamei}', header=True, index=True)

            else:
                with pd.ExcelWriter(f'{saveto}/pdp.xlsx', mode='w') as writer:
                    pdp_df.to_excel(writer, sheet_name=f'{xnamei}', header=True, index=True)

        
        aic,bic = gam_abic(gam_model,y,X)
        print('AIC: ',aic)
        print('BIC: ',bic)
        
        file.write(f'AIC: {aic}')
        file.write('\n')
        file.write(f'BIC: {bic}')
        file.write('\n')
        file.close()
        with open(f'{saveto}/gam__{disease}_{yname}_lag{lagnum}','wb') as f:
            pickle.dump(gam_model,f)
        with open(f'{saveto}/gam__{disease}_{yname}_lag{lagnum}','wb') as f:
            pickle.dump(gam_model,f)
        return gam_model


diseases = ['Dengue Fever','HFMD','Campylobacter enteritis',
'Salmonellosis(non-enteric fevers)','Acute Upper Respiratory Tract infections',
'Acute Conjunctivitis', 'Acute Diarrhoea']    
 
ytype = ['Incidence','Incidence_Rate']
pool = multiprocessing.Pool(processes=60)  # Use the number of CPU cores

# # Use multiprocessing to parallelize the execution of incidence_model
pool.starmap(poissongam, [(ytype,disease) for disease in diseases for ytype in ['Incidence','Incidence_Rate']])

# Close the pool to free up resources
pool.close()
pool.join()

pool = multiprocessing.Pool(processes=40)  # Use the number of CPU cores
pool.starmap(poissongamlag,[(t,disease,l) for disease in diseases for t in ytype for l in range(9)])
pool.close()
pool.join()

