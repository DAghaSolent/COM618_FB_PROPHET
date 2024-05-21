import pandas as pd

pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)

df = pd.read_excel(r'alcoholspecificdeaths2021.xlsx', sheet_name='Table 1', skiprows=4)

print(df.head())