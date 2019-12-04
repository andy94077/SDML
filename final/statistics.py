import sys
import numpy as np
import pandas as pd
import seaborn as sns

df = pd.read_csv(sys.argv[1]).drop(columns=['Id'])
pd.set_option('display.max_columns', 50)
pd.set_option('display.width', 1000)
pd.set_option('precision', 3)

sns.set(font_scale=0.7)
print('df.describe()')
print(df[df.columns[:-1]].describe())
melt = pd.melt(df[df.columns[:-1]])
melt.columns = ['dimension', 'value']
melt.drop(np.arange(melt.shape[0])[melt['value']>20], inplace=True)
box = sns.boxplot(x='dimension', y='value', data=melt)#, showfliers=False)
box.get_figure().savefig('box.jpg', bbox_inches='tight', dpi=300)

print('\ndf.corr()')
print(df.corr())
heatmap = sns.heatmap(df.corr())
heatmap.get_figure().savefig('heatmap.jpg', bbox_inches='tight', dpi=300)

g = df.groupby(['Class'])
print('\ng.mean()')
print(g.mean())

print('\ng.var()')
print(g.var())

print('\ng.count()')
print(g.count())
