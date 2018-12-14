import datetime as dt
import pandas as pd
import csv

df = pd.read_csv("./Processed/HK0700.csv", parse_dates=[[0,1]], infer_datetime_format=True, date_parser=lambda x: pd.datetime.strptime(x, '%Y-%m-%D, %H:%M'))

df = df[((pd.to_datetime("2015-01-01") >= df.mydate))]
df.to_csv('./Processed/HK0700_p.csv')

#with open('./Processed/HK0007.csv', 'b') as csvfile:
#    reader = csv.reader(csvfile, delimiter = ',')
