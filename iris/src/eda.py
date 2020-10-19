import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt 
import seaborn as sns
iris = sns.load_dataset('iris')

sns.pairplot(iris, hue='species', palette='Dark2')
#plt.savefig('../imgs/iris_pairplot.png')
plt.show();

setosa = iris[iris['species']=='setosa']
sns.kdeplot( setosa['sepal_width'], setosa['sepal_length'],
                 cmap="plasma", shade=True, shade_lowest=False)
plt.savefig('../imgs/setosa_kdeplot.png')
plt.show();