import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
data=sns.load_dataset('iris')
print(data.info())
print(data.describe())
print(data['species'].value_counts())
sns.histplot(data['sepal_length'],kde=True)
plt.title("Distribution of sepal length")
plt.show()
sns.boxplot(x='species',y='sepal_length',data=data)
plt.title("sepal length by species")
plt.show()
sns.violinplot(x='species',y='sepal_length',data=data)
plt.title("petal length distribution by species")
plt.show()
sns.pairplot(data,hue='species')
plt.title("pairwise comparision of iris features",y=1.02)
plt.show()