import pandas as pd
import os
from datetime import datetime
import convert
import numpy as np
from sklearn.preprocessing import StandardScaler
from dateutil.relativedelta import relativedelta
from create_dirs import *
# clean and merge diseases data
df = pd.ExcelFile(f'{data_dir}/2021-2012.xlsx')
sheet_names = df.sheet_names
sheets = []
# format the sheet of 2022
sheet_2022 = pd.read_excel(f'{data_dir}/2021-2012.xlsx', sheet_name='2022',header = None)
sheet_2022.columns = sheet_2022.iloc[1]
sheet_2022 = sheet_2022.iloc[2:]
weekdate = sheet_2022['Start to End']
start = []
end = []
for wd in weekdate:
    temp = wd.split(' - ')
    start.append(temp[0])
    end.append(temp[1])
column1 = sheet_2022.iloc[:,0]
sheet_2022 = sheet_2022.iloc[:,3:]
sheet_2022.insert(0,'Epidemiology Wk',column1)
sheet_2022.insert(1,'Start',start)
sheet_2022.insert(2,'End',end)
sheets.append(sheet_2022)
# format rest sheets
for sheet_name in sheet_names[1:]:
    # Read the data from the sheet into a DataFrame
    sheet_data = pd.read_excel(f'{data_dir}/2021-2012.xlsx', sheet_name=sheet_name,header = None)
    sheet_data.columns = sheet_data.iloc[1]
    sheet_data = sheet_data.iloc[2:]
    sheets.append(sheet_data)
# merge data
sheets.reverse()
diseases = pd.concat(sheets)
# index
index_c = list(range(len(diseases)))
diseases.index = index_c
# drop rows with all NA values
sub_diseases = diseases.iloc[:,3:]
na_rows = sub_diseases[sub_diseases.isna().all(axis=1)]
diseases = diseases.drop(na_rows.index)
# some diseases counts are average daily number, we convert it to weekly basis
column_daily = ['Acute Upper Respiratory Tract infections',
                'Acute Conjunctivitis',
                'Acute Diarrhoea',
                'Chickenpox']
for c in column_daily:
    diseases[c] = diseases[c] * 7
# convert start/end date to datatime format
for i in range(len(diseases)):
    if type(diseases['Start'][i])==str:
        temp=diseases['Start'][i].replace(" ","")
        diseases['Start'][i] = datetime.strptime(temp,"%d/%m/%Y")
    if type(diseases['End'][i])==str:
        temp = diseases['End'][i].replace(" ", "")
        diseases['End'][i] = datetime.strptime(temp,"%d/%m/%Y")
diseases_years = []
for d in diseases['Start']:
    diseases_years.append(d.year)
index_c = []
for i in range(len(diseases)):
    if (diseases['Start'][i].month == 12) & (diseases['Epidemiology Wk'][i]==1):
        index_c.append(f"{diseases_years[i]+1}_{diseases['Epidemiology Wk'][i]}")
    elif (diseases['Start'][i].month == 1) & (diseases['Epidemiology Wk'][i]==53):
        index_c.append(f"{diseases_years[i] - 1}_{diseases['Epidemiology Wk'][i]}")
    elif (diseases['Start'][i].month == 1) & (diseases['Epidemiology Wk'][i] == 52):
        index_c.append(f"{diseases_years[i] - 1}_{diseases['Epidemiology Wk'][i]}")
    else:
        index_c.append(f"{diseases_years[i]}_{diseases['Epidemiology Wk'][i]}")
diseases.index = index_c

# clean climate data
climate_daily = pd.read_csv(f'{data_dir}/Climate_2012to2022.csv',
                            header=0,index_col=None,sep=',')
years = []
index_c = []
for i in range(len(climate_daily['dateV'])):
    temp = climate_daily['dateV'][i]
    if type(temp)==str:
        climate_daily['dateV'][i] = datetime.strptime(temp,"%d-%m-%y")
    years.append(climate_daily['dateV'][i].year)

    if (climate_daily['dateV'][i].month == 12) & (climate_daily['epiweekV'][i] == 1):
        index_c.append(f"{years[i] + 1}_{climate_daily['epiweekV'][i]}")
    elif (climate_daily['dateV'][i].month == 1) & (climate_daily['epiweekV'][i] == 53):
        index_c.append(f"{years[i] - 1}_{climate_daily['epiweekV'][i]}")
    elif (climate_daily['dateV'][i].month == 1) & (climate_daily['epiweekV'][i] == 52):
        index_c.append(f"{years[i] - 1}_{climate_daily['epiweekV'][i]}")
    else:
        index_c.append(f"{years[i]}_{climate_daily['epiweekV'][i]}")
# group the dataset by year_week and average the values
columns_to_average = climate_daily.columns[2:]
climate_daily['yearWeek'] = index_c
grouped = climate_daily.groupby('yearWeek',sort=False)
climate = grouped[columns_to_average].mean()

# clean weather data
weather_raw = pd.read_csv(f'{data_dir}/weather.csv',header=0,index_col=None)

columns_to_average = weather_raw.columns[7:]
for i in range(len(weather_raw)):
    temp = weather_raw['Date'][i]
    if type(temp)==str:
        weather_raw['Date'][i] = datetime.strptime(temp,"%d/%m/%Y")
# we just want the data after 2012
weather = weather_raw[weather_raw['Year'] >= 2012]
index_c = list(range(len(weather)))
weather.index = index_c

yearWeek = []
for d in weather['Date']:
    filtered_df = climate_daily[climate_daily['dateV'] == d]
    cell_value = filtered_df.iloc[0, filtered_df.columns.get_loc('yearWeek')]
    yearWeek.append(cell_value)

weather['yearWeek'] = yearWeek

weather = weather.drop(columns = ['Date','Time','Year','Month','Day','DOW','DOW.Name'])
grouped_weather = weather.groupby('yearWeek',sort=False)
weather = grouped_weather[columns_to_average].mean()

# Merge the three DataFrames by index using pd.concat()
merged_df = pd.concat([diseases,climate,weather], axis=1)
years = []
for i in range(len(merged_df)):
    years.append(merged_df.index[i][0:4])
merged_df.insert(1,'Year',years)

pop = pd.read_csv(f'{data_dir}/M810001.csv',header = 0, index_col=0)
pop_ls = []
for index, row in merged_df.iterrows():
    yr = row['Year']
    pop_ls.append(pop.loc[yr,'Total Population'])
logpop = np.log(pop_ls)
merged_df['pop'] = pop_ls
merged_df['logpop'] = logpop
merged_df['year_week'] = merged_df['Year'].astype(str) + '-' + merged_df['Epidemiology Wk'].astype(str)

# Convert 'year_week' to datetime format
merged_df['year_week'] = pd.to_datetime(merged_df['year_week'] + '-0', format='%Y-%U-%w')
merged_df.set_index('year_week',inplace=True)
merged_df.to_csv(f'{data_dir}/merged.csv',header=True,index=True)