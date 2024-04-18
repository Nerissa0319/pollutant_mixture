import numpy as  np
from scipy.special import gamma
from scipy.stats import poisson
def gp_aic(model,Y,X):
    pred_mean,pred_var = model.predict_y(X)
    log_likelihood1 = np.sum(poisson.logpmf(Y[:,0],pred_mean[:,0]))
    log_likelihood2 = []
    for i in range(len(Y)):
        t1 = Y[:,0][i]*np.log(pred_mean[:,0][i])
        t2 = t1 - pred_mean[:,0][i]
        t3 = gamma(Y[:,0][i] + 1)
        t4 = np.log(t3)
        l = t2-t4
        log_likelihood2.append(l)
    log_likelihood2 = sum(log_likelihood2)
    log_likelihood = max(log_likelihood2,log_likelihood1)
    k = len(model.parameters)
    n = len(model.data[1])
    aic = 2 * k - 2 * log_likelihood
    return aic    

def gp_bic(model,Y,X):
    k = len(model.parameters)
    pred_mean,pred_var = model.predict_y(X)
    log_likelihood1 = np.sum(poisson.logpmf(Y[:,0],pred_mean[:,0]))
    log_likelihood2 = []
    for i in range(len(Y)):
        t1 = Y[:,0][i]*np.log(pred_mean[:,0][i])
        t2 = t1 - pred_mean[:,0][i]
        t3 = gamma(Y[:,0][i] + 1)
        t4 = np.log(t3)
        l = t2-t4
        log_likelihood2.append(l)
    log_likelihood2 = sum(log_likelihood2)
    log_likelihood = max(log_likelihood2,log_likelihood1)
    n = len(model.data[1])
    bic = -2 * log_likelihood + np.log(n) * k
    return bic

# def gp_abic(model,Y,X):
#     k = len(model.parameters)
#     pred_mean,pred_var = model.predict_y(X)
#     logll = model.


def glm_abic(model,Y,X):
    pred_mean = model.predict(X).values
    log_likelihood1 = np.sum(poisson.logpmf(Y,pred_mean))
    log_likelihood2 = []
    for i in range(len(Y)):
        t1 = Y[i]*np.log(pred_mean[i])
        t2 = t1 - pred_mean[i]
        t3 = gamma(Y[i] + 1)
        t4 = np.log(t3)
        l = t2-t4
        log_likelihood2.append(l)
    log_likelihood2 = sum(log_likelihood2)
    log_likelihood = max(log_likelihood2,log_likelihood1)
    k = X.shape[1]
    n = len(Y)
    aic = 2 * k - 2 * log_likelihood
    bic = -2 * log_likelihood + np.log(n) * k
    return aic, bic

def gam_abic(model,Y,X):
    pred_mean = model.predict(X)
    log_likelihood1 = np.sum(poisson.logpmf(Y,pred_mean))
    log_likelihood2 = []
    for i in range(len(Y)):
        t1 = Y[i]*np.log(pred_mean[i])
        t2 = t1 - pred_mean[i]
        t3 = gamma(Y[i] + 1)
        t4 = np.log(t3)
        l = t2-t4
        log_likelihood2.append(l)
    log_likelihood2 = sum(log_likelihood2)
    log_likelihood = max(log_likelihood2,log_likelihood1)
    k = X.shape[1]
    n = model.statistics_['edof']
    aic = 2 * k - 2 * log_likelihood
    bic = -2 * log_likelihood + np.log(n) * k
    return aic, bic