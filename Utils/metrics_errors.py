# -*- coding: utf-8 -*-
"""
@author: Juan Gonzalez Sopena

Metrics to measure the accuracy of deterministic forecasts and prediction intervals
"""

import numpy as np
import pandas as pd

df = pd.DataFrame(columns = ['actual', 'prediction', 'prediction_ref'])
df['actual'] = ['10','8','5','7','8']
df['prediction'] = ['9','8','7','6','6']
df['prediction_ref'] = ['7','6','7','5','5']
df = df.astype(float)

df = pd.DataFrame(columns = ['actual', 'prediction'])
df['actual'] = [4,0,2,0,1]
df['prediction'] = [3,1,2,1,1]
#df = df.astype(float)

N = len(df['actual'])

pi = pd.DataFrame(columns = ['actual', 'lower', 'upper'])
pi['actual'] = [5,6,9,6,5]
pi['lower'] = [4,5,6,7,5]
pi['upper'] = [9,9,8,9,7]
pi = pi.astype(float)

""""""""""""""""""""""""
"   Determ forecasts   "
""""""""""""""""""""""""

def mape(prediction, actual):
    """
    Function to compute MAPE (mean absolute percentage error)
    When actual = 0, when a time step is zero, it's not computed
    Inputs: 
        prediction: set of predictions 
        actual: set of real values
    Output: mape (%)
    """
    mask = actual != 0
    return (np.fabs(actual - prediction)/actual)[mask].mean() *100

def mape2(prediction, actual):
    """
    Function to compute MAPE (mean absolute percentage error)
    When actual = 0, when a time step is zero, it's not computed
    Inputs: 
        prediction: set of predictions 
        actual: set of real values
    Output: mape (%)
    Note (12122019): recheck by Bidisha
    """
    df = pd.DataFrame()
    df['actual'] = actual
    df['prediction'] = prediction
    
    df['ape'] = np.fabs(df['actual']-df['prediction'])/df['actual']
    mask = df['ape'] < 1 
    return (np.fabs(actual - prediction)/actual)[mask].mean()*100 


def mae(prediction, actual, total_p, norm):
    """
    Function to compute MAE (mean absolute error) or NMAE
    Inputs: 
        prediction: set of predictions 
        actual: set of real values
        total_p: installed capacity
        norm: normalize the metric        
    Output: mae (abs. value), nmae (%)
    """
    value = (abs(prediction - actual))
    value = value.sum()
    if norm == True:
        value = (value/(len(prediction)*total_p))*100
    if norm == False: 
        value = (value/(len(prediction)))
    return value

def rmse(prediction, actual, total_p, norm = True):
    """
    Function to compute RMSE (root-mean-square error) or NRMSE
    Inputs: 
        prediction: set of predictions 
        actual: set of real values
        total_p: installed capacity
        norm: normalize the metric   
    Output: nmae (%)
    """
    value = (abs(prediction - actual))**2
    value = value.sum()
    value = np.sqrt(value/len(prediction))
    if norm == True:
        value = (value/total_p)*100
    if norm == False:
        value = value
    return value

def bias(prediction, actual, total_p, norm):
    """
    Function to compute bias
    Inputs: 
        prediction: set of predictions 
        actual: set of real values
    Output: bias
    """
    value = actual - prediction
    value = value.sum()
    value = value/len(prediction)
    if norm == True:
        value = (value/total_p)*100
    if norm == False:
        value = value
    return value

def sde(prediction, actual, total_p, norm):
    """
    Function to compute SDE (standard deviation of errors)
    Inputs: 
        prediction: set of predictions 
        actual: set of real values
    Output: bias
    """
    def residue_mean(prediction, actual):
        res = actual - prediction
        res_mean = res.mean()
        return res_mean
        
    #res = actual - prediction
    res_mean = residue_mean(prediction, actual)
    value = (abs(actual - prediction) - res_mean)**2 
    value = value.sum()
    value = np.sqrt(value/len(prediction))
    if norm == True:
        value = (value/total_p)*100
    if norm == False:
        value = value
    return value

def index_agreement(prediction, actual):
    """
    Function to compute IA (index of agreement)
    Inputs: 
        prediction: set of predictions 
        actual: set of real values
    Output: bias
    Note: there are refined versions of this index by the author (original Wilmott 1981)
    """
    value_num = (abs(prediction - actual))**2
    value_num = value_num.sum() 
    value_den = (abs(prediction - actual.mean()) + abs(actual - actual.mean()))**2
    value_den = value_den.sum()
    value = 1 - (value_num/value_den)
    return value

def skill_score(metric, prediction, prediction_ref, actual):
    """
    Function to compute skill score
    Input:
        metric: metric selected to be compared (mae, rmse or mape)
        prediction: predictions of proposed model
        prediction_ref: predictions of benchmark model
        actual: actual values
    Output:
        Skill score (%)
    Note: Skill score is negative if proposed model performs worst than benchmark
    """
    def mae(prediction, actual):
        value = (abs(prediction - actual))
        value = value.sum() 
        value = (value/(len(prediction)))
        return value
    
    def rmse(prediction, actual):
        value = (abs(prediction - actual))**2
        value = value.sum()
        value = np.sqrt(value/len(prediction))
        return value
    
    def mape(prediction, actual):
        value = (abs(prediction - actual) / actual)
        value = value.sum()
        value = (value/len(prediction))*100
        return value

    if metric == 'mae':
        mae_prop = mae(prediction, actual)
        mae_ref =  mae(prediction_ref, actual)
        skill_score = (1 - (mae_prop/mae_ref))*100
        return skill_score
    
    if metric == 'rmse':
        rmse_prop = rmse(prediction, actual)
        rmse_ref =  rmse(prediction_ref, actual)
        skill_score = (1 - (rmse_prop/rmse_ref))*100
        return skill_score
    
    if metric == 'mape':
        mape_prop = mape(prediction, actual)
        mape_ref =  mape(prediction_ref, actual)
        skill_score = (1 - (mape_prop/mape_ref))*100
        return skill_score
    else:
        print("choose a metric: mae, rmse or mape")

 
def point_metrics(prediction, actual, total_p, norm):
    metrics = pd.DataFrame(columns=['METRIC', 'VALUE'])
    metrics['METRIC'] = ['MAE', 'RMSE', 'MAPE', 'BIAS', 'SDE', 'IA']      
    mae_ = mae(prediction, actual, total_p, norm)
    rmse_ = rmse(prediction, actual, total_p, norm)
    mape_ = mape2(prediction, actual)
    bias_ = bias(prediction, actual, total_p, norm)
    sde_ = sde(prediction, actual, total_p, norm)
    ia_ = index_agreement(prediction, actual)
    metrics['VALUE'] = [mae_, rmse_, mape_, bias_, sde_, ia_] 
    return metrics

"""
Deal with missing values
"""

def mae_with_nans(prediction, actual, total_p, norm):
    """
    Function to compute MAE (mean absolute error) or NMAE
    Inputs: 
        prediction: set of predictions (pandas Series or numpy array)
        actual: set of real values (pandas Series or numpy array)
        total_p: installed capacity
        norm: normalize the metric (boolean)      
    Output: mae (abs. value), nmae (%)
    """
    # Ensure the inputs are pandas Series
    prediction = pd.Series(prediction)
    actual = pd.Series(actual)
    
    # Drop NaN values
    valid_mask = prediction.notna() & actual.notna()
    prediction = prediction[valid_mask]
    actual = actual[valid_mask]
    
    # Compute absolute errors
    abs_errors = abs(prediction - actual)
    
    # Calculate the mean absolute error or normalized mean absolute error
    if norm:
        value = (abs_errors.sum() / (len(prediction) * total_p)) * 100
    else:
        value = abs_errors.mean()
    
    return value
    
def mape2_with_nans(prediction, actual):
    """
    Function to compute MAPE (mean absolute percentage error)
    When actual = 0, or a time step is zero or NaN, it's not computed.
    Inputs: 
        prediction: set of predictions (pandas Series or numpy array)
        actual: set of real values (pandas Series or numpy array)
    Output: mape (%)
    """
    # Convert inputs to pandas Series to ensure compatibility
    prediction = pd.Series(prediction)
    actual = pd.Series(actual)

    # Drop NaN values in both prediction and actual
    valid_mask = prediction.notna() & actual.notna() & (actual != 0)  # Remove NaNs and actuals that are 0
    prediction = prediction[valid_mask]
    actual = actual[valid_mask]
    
    # Calculate absolute percentage error (APE)
    ape = np.abs(actual - prediction) / actual
    
    # Filter out APEs greater than 1 (if required by the business logic)
    ape = ape[ape < 1]
    
    # Return the mean of valid APEs as MAPE percentage
    return ape.mean() * 100
    
def rmse_with_nans(prediction, actual, total_p, norm=True):
    """
    Function to compute RMSE (root-mean-square error) or NRMSE
    Inputs: 
        prediction: set of predictions (pandas Series or numpy array)
        actual: set of real values (pandas Series or numpy array)
        total_p: installed capacity
        norm: normalize the metric (boolean)   
    Output: rmse (abs. value), nrmse (%)
    """
    prediction = pd.Series(prediction)
    actual = pd.Series(actual)
    
    valid_mask = prediction.notna() & actual.notna()
    prediction = prediction[valid_mask]
    actual = actual[valid_mask]
    
    mse = ((prediction - actual) ** 2).mean()
    value = np.sqrt(mse)
    
    if norm:
        value = (value / total_p) * 100
    
    return value

def mse_with_nans(prediction, actual, total_p, norm=True):
    """
    Function to compute MSE (mean-square error) or NMSE
    Inputs: 
        prediction: set of predictions (pandas Series or numpy array)
        actual: set of real values (pandas Series or numpy array)
        total_p: installed capacity
        norm: normalize the metric (boolean)   
    Output: mse (abs. value), nmse (%)
    """
    prediction = pd.Series(prediction)
    actual = pd.Series(actual)
    
    valid_mask = prediction.notna() & actual.notna()
    prediction = prediction[valid_mask]
    actual = actual[valid_mask]
    
    value = ((prediction - actual) ** 2).mean()
    
    if norm:
        value = (value / total_p) * 100
    
    return value

def bias_with_nans(prediction, actual, total_p, norm):
    """
    Function to compute bias
    Inputs: 
        prediction: set of predictions (pandas Series or numpy array)
        actual: set of real values (pandas Series or numpy array)
        total_p: installed capacity
        norm: normalize the metric (boolean)   
    Output: bias (abs. value), normalized bias (%)
    """
    prediction = pd.Series(prediction)
    actual = pd.Series(actual)
    
    valid_mask = prediction.notna() & actual.notna()
    prediction = prediction[valid_mask]
    actual = actual[valid_mask]
    
    bias_value = (actual - prediction).mean()
    
    if norm:
        bias_value = (bias_value / total_p) * 100
    
    return bias_value

def sde_with_nans(prediction, actual, total_p, norm):
    """
    Function to compute SDE (standard deviation of errors)
    Inputs: 
        prediction: set of predictions (pandas Series or numpy array)
        actual: set of real values (pandas Series or numpy array)
        total_p: installed capacity
        norm: normalize the metric (boolean)   
    Output: sde (abs. value), normalized sde (%)
    """
    prediction = pd.Series(prediction)
    actual = pd.Series(actual)
    
    valid_mask = prediction.notna() & actual.notna()
    prediction = prediction[valid_mask]
    actual = actual[valid_mask]
    
    residuals = actual - prediction
    residual_mean = residuals.mean()
    sde_value = np.sqrt(((residuals - residual_mean) ** 2).mean())
    
    if norm:
        sde_value = (sde_value / total_p) * 100
    
    return sde_value

def index_agreement_with_nans(prediction, actual):
    """
    Function to compute IA (index of agreement)
    Inputs: 
        prediction: set of predictions (pandas Series or numpy array)
        actual: set of real values (pandas Series or numpy array)
    Output: index of agreement
    Note: there are refined versions of this index by the author (original Wilmott 1981)
    """
    prediction = pd.Series(prediction)
    actual = pd.Series(actual)
    
    valid_mask = prediction.notna() & actual.notna()
    prediction = prediction[valid_mask]
    actual = actual[valid_mask]
    
    numerator = ((prediction - actual) ** 2).sum()
    denominator = ((abs(prediction - actual.mean()) + abs(actual - actual.mean())) ** 2).sum()
    
    ia_value = 1 - (numerator / denominator)
    
    return ia_value
    
def point_metrics_with_nans(prediction, actual, total_p, norm):
    metrics = pd.DataFrame(columns=['METRIC', 'VALUE'])
    metrics['METRIC'] = ['MAE', 'RMSE','MSE', 'MAPE', 'BIAS', 'SDE', 'IA']      
    mae_ = mae_with_nans(prediction, actual, total_p, norm)
    rmse_ = rmse_with_nans(prediction, actual, total_p, norm)
    mse_ = mse_with_nans(prediction, actual, total_p, norm)
    mape_ = mape2_with_nans(prediction, actual)
    bias_ = bias_with_nans(prediction, actual, total_p, norm)
    sde_ = sde_with_nans(prediction, actual, total_p, norm)
    ia_ = index_agreement_with_nans(prediction, actual)
    metrics['VALUE'] = [mae_, rmse_, mse_, mape_, bias_, sde_, ia_] 
    return metrics


def picp_with_nans(actual, lower, upper):
    """
    Function to calculate the PI coverage probability (PICP) of a prediction interval.
    Inputs:
        actual: actual values
        lower: lower boundary of the prediction interval
        upper: upper boundary of the prediction interval
    Output:
        PICP (%)
    """
    df = pd.DataFrame()
    df['actual'] = actual
    df['lower'] = lower
    df['upper'] = upper
    
    # Remove rows with NaNs in any of the columns
    valid_mask = df['actual'].notna() & df['lower'].notna() & df['upper'].notna()
    df = df[valid_mask]
    
    # Calculate PICP
    value = len(df.loc[(df['actual'] <= df['upper']) & (df['actual'] >= df['lower'])]) / len(df)
    value = value * 100
    return value

def mpiw_with_nans(actual, lower, upper, norm, max_value, min_value=0):
    """
    Function to calculate the mean prediction interval width (MPIW) 
    (also normalized) of a prediction interval.
    Note: In the literature is common to find the normalized version of 
    this metric as PINAW (PI normalized average width).
    Inputs:
        actual: actual values
        lower: lower boundary of the prediction interval
        upper: upper boundary of the prediction interval
        norm: whether value is normalized or not
        max_value: max value of wind power output
        min_value: minimum value of wind power output (zero)    
    Output:
        MPIW or NMPIW/PINAW (%)
    """
    df = pd.DataFrame()
    df['actual'] = actual
    df['lower'] = lower
    df['upper'] = upper
    
    # Remove rows with NaNs in the lower and upper bounds
    valid_mask = df['lower'].notna() & df['upper'].notna()
    df = df[valid_mask]
    
    # Calculate MPIW or normalized MPIW
    summ = (df['upper'] - df['lower']).sum()
    if norm:
        value = (summ / ((max_value - min_value) * len(df))) * 100
    else:
        value = summ / len(df)
    return value

""""""""""""""""""""""""
" Prediction intervals OLD "
""""""""""""""""""""""""
    
def picp(actual, lower, upper):
    """
    Function to calculate the PI coverage probability (PICP) of a prediction interval.
    Inputs:
        actual: actual values
        lower: lower boundary of the prediction interval
        upper: upper boundary of the prediction interval
    Output:
        PICP (%)
    """
    df = pd.DataFrame()
    df['actual'] = actual
    df['lower'] = lower
    df['upper'] = upper
       
    value = len(df.loc[(df['actual'] <= df['upper']) & (df['actual'] >= df['lower'])])/df.shape[0]
    value = value*100
    return value

def mpiw(actual, lower, upper, norm, max_value, min_value = 0):
    """
    Function to calculate the mean prediction interval width (MPIW) 
    (also normalized) of a prediction interval.
    Note: In the literature is common to find  the normalized version of 
    this metric as PINAW (PI normalized average width).
    Inputs:
        actual: actual values
        lower: lower boundary of the prediction interval
        upper: upper boundary of the prediction interval
        norm: whether value is normalized or not
        max_value: max value of wind power output
        min_value: minimum value of wind power output (zero)    
    Output:
        MPIW or NMPIW/PINAW (%)
    """
    df = pd.DataFrame()
    df['actual'] = actual
    df['lower'] = lower
    df['upper'] = upper
    
    summ = (df['upper'] - df['lower']).sum()
    if norm == True:
        value = ((summ)/((max_value - min_value) * df.shape[0]))*100
    if norm == False:
        value = ((summ)/(df.shape[0]))
    return value

def pinaw(actual, lower, upper, max_value, min_value = 0):
    """
    Function to calculate PINAW of a prediction interval.
    Note: MPIW not possible to calculate
    Inputs:
        actual: actual values
        lower: lower boundary of the prediction interval
        upper: upper boundary of the prediction interval
        norm: whether value is normalized or not
        max_value: max value of wind power output
        min_value: minimum value of wind power output (zero)    
    Output:
        PINAW (%)
    """
    df = pd.DataFrame()
    df['actual'] = actual
    df['lower'] = lower
    df['upper'] = upper
    
    summ = (df['upper'] - df['lower']).sum()
    value = ((summ)/((max_value - min_value) * df.shape[0]))*100
    return value

def cwc(actual, lower, upper, alpha, eta, max_value, min_value = 0):
    """
    Function to calculate the coverage width-based criterion (CWC) of a prediction interval.
    (More info on PIs for Short-Term Wind Farm Power Generation Forecasts, Khosravi 2013)
    Input:
        actual: actual values
        lower: lower boundary of the prediction interval
        upper: upper boundary of the prediction interval
        mu: hyperparameter, in practice 1-alpha
        eta: hyperparameter, usually value between 50-100 to penalize invalid PIs
        max_value: max value of wind power output
        min_value: minimum value of wind power output (zero)
    Output:
        CWC
    """   
    def picp(df):
        value = len(df.loc[(df['actual'] <= df['upper']) & (df['actual'] >= df['lower'])])/df.shape[0]
        value = value
        return value

    def pinaw(df, max_value, min_value = 0):
        summ = (df['upper'] - df['lower']).sum()
        value = (summ)/((max_value - min_value) * df.shape[0])
        return value    
    
    df = pd.DataFrame()
    df['actual'] = actual
    df['lower'] = lower
    df['upper'] = upper
    
    picp_ = picp(df)
    pinaw_ = pinaw(df, max_value = max_value)
    mu = 1 - alpha
    if picp_ >= mu:
        value = pinaw_
    if picp_ < mu:
        value = pinaw_ * (1 + np.exp(-eta*(picp_ - mu)))
    return value    

#cwc(actual = pi['actual'], 
#    lower = pi['lower'], 
#    upper = pi['upper'], 
#    mu = 0.95, 
#    eta = 50, 
#    max_value = 10)

def ace(actual, lower, upper, alpha):
    """
    Function to calculate the average coverage error (ACE) of a prediction interval.
    Input:
        actual: actual values
        lower: lower boundary of the prediction interval
        upper: upper boundary of the prediction interval
        alpha: sign. level of the prediction interval
    Output:
        ACE
    """
    df = pd.DataFrame()
    df['actual'] = actual
    df['lower'] = lower
    df['upper'] = upper
    
    picp = len(df.loc[(df['actual'] <= df['upper']) & (df['actual'] >= df['lower'])])/df.shape[0]
    value = picp - (1 - alpha)
    return value 

def int_sharp(actual, lower, upper, alpha):
    """
    Function to calculate the interval sharpness (IS) of a prediction interval.
    Input:
        actual: actual values
        lower: lower boundary of the prediction interval
        upper: upper boundary of the prediction interval
        alpha: sign. level of the prediction interval
    Output:
        Interval sharpness
    """
    df = pd.DataFrame()
    df['actual'] = actual
    df['lower'] = lower
    df['upper'] = upper
    
    df.loc[ df['actual'] < df['lower'], 'sct' ] = -2*alpha*(df['upper']-df['lower']) - 4*abs(df['lower']-df['actual'])
    df.loc[ (df['actual'] >= df['lower']) & (df['actual'] <= df['upper']), 'sct' ] = -2*alpha*(df['upper']-df['lower'])
    df.loc[ df['actual'] > df['upper'], 'sct' ] = -2*alpha*(df['upper']-df['lower']) - 4*abs(df['actual']-df['upper'])
    value = df['sct'].sum()
    value = value/df.shape[0]
    return value

def int_sharp_i(actual, lower, upper, alpha):
    """
    Function to calculate the interval sharpness (IS) of a prediction interval.
    Input:
        actual: actual values
        lower: lower boundary of the prediction interval
        upper: upper boundary of the prediction interval
        alpha: sign. level of the prediction interval
    Output:
        Interval sharpness
    """
    df = pd.DataFrame()
    df['actual'] = actual
    df['lower'] = lower
    df['upper'] = upper
    
    df.loc[ df['actual'] < df['lower'], 'sct' ] = -2*alpha*(df['upper']-df['lower']) - 4*abs(df['lower']-df['actual'])
    df.loc[ (df['actual'] >= df['lower']) & (df['actual'] <= df['upper']), 'sct' ] = -2*alpha*(df['upper']-df['lower'])
    df.loc[ df['actual'] > df['upper'], 'sct' ] = -2*alpha*(df['upper']-df['lower']) - 4*abs(df['actual']-df['upper'])   
    return df['sct']

def interval_metrics(actual, lower, upper, alpha, eta, max_value, min_value = 0):
    metrics = pd.DataFrame(columns=['METRIC', 'VALUE'])
    metrics['METRIC'] = ['PICP', 'PINAW', 'CWC', 'ACE', 'IS']      
    picp_ = picp(actual, lower, upper)
    pinaw_ = pinaw(actual, lower, upper, max_value, min_value = 0)
    cwc_ = cwc(actual, lower, upper, alpha, eta, max_value, min_value = 0)
    ace_ = ace(actual, lower, upper, alpha)
    is_ = int_sharp(actual, lower, upper, alpha)
    
    metrics['VALUE'] = [picp_, pinaw_, cwc_, ace_, is_] 
    return metrics

#int_metrics = interval_metrics(actual = pi['actual'],
#                               lower = pi['lower'],
#                               upper = pi['upper'],
#                               alpha = 0.05,
#                               eta = 50,
#                               max_value = 10)
                              





