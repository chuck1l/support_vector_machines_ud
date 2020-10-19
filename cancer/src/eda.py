import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_breast_cancer

cancer = load_breast_cancer()
print(cancer.keys())
print(cancer['DESCR'])
print(cancer.feature_names)

df_feat = pd.DataFrame(cancer['data'], columns=cancer['feature_names'])
print(df_feat.info())
df_target = pd.DataFrame(cancer['target'], columns=['Cancer'])
print(df_feat.head())
df = pd.concat([df_feat, df_target], axis=1)
print(df.head())
sns.pairplot(df, palette='Dark2')
plt.savefig('../imgs/cancer_pairplot.png');