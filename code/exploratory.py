import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from create_dirs import *
df = pd.read_csv(f'{data_dir}/merged.csv',header=0,index_col=0)
diseases = ['Dengue Fever','HFMD','Campylobacter enteritis',
'Salmonellosis(non-enteric fevers)','Acute Upper Respiratory Tract infections',
'Acute Conjunctivitis', 'Acute Diarrhoea']  
# since the weather data has no records after 2020, we drop the rows for that
df = df[0:417]
# we take Dengue Fever as the dependant variable, so we drop other columns
for d in diseases:
    columns_to_explore = [d,'MeanT','AH','pm10','pm25','o3','no2','so2','co','Rain','logpop','pop']
    disease_df = df[columns_to_explore]

    # exploratory data analysis
    # 1. inspect the data
    # Display the first few rows of the DataFrame
    print(disease_df.head())
    # Check the dimensions of the DataFrame (number of rows, number of columns)
    print(disease_df.shape)
    # Get summary statistics of the numerical variables
    print(disease_df.describe())
    # # Check the data types of each column
    # print(disease_df.dtypes)
    # # Check for missing values
    # print(disease_df.isnull().sum())
    disease_df['Incidence_Rate'] = disease_df[d]/disease_df['pop']
    disease_df.to_csv(f'{data_dir}/{d}.csv',header=True,index=True)

    lag_range = range(8)
    disease_df1 = disease_df.copy()
    independent_vars = [d,'Incidence_Rate','MeanT','AH','pm10','pm25','o3','no2','so2','co','Rain']
    for var in independent_vars:
        for l in lag_range:
            col_name = var + '_lag{}'.format(l+1)
            disease_df1[col_name] = disease_df1[var].shift(l+1)
    disease_df1 = disease_df1.dropna()
    disease_df1.to_csv(f'{data_dir}/{d}_lag.csv',header=True,index=True)