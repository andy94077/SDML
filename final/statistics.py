import sys
import pandas as pd

df = pd.read_csv(sys.argv[1]).drop(columns=['Id'])
pd.set_option('display.max_columns', 50)
pd.set_option('display.width', 1000)
pd.set_option('precision', 3)

print('df.describe()')
print(df[df.columns[:-1]].describe())

g = df.groupby(['Class'])
print('\ng.mean()')
print(g.mean())

print('\ng.var()')
print(g.var())

print('\ng.count()')
print(g.count())
