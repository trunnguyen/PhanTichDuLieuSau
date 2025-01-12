import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

iris_df = pd.read_csv(r"C:\Users\nguye\OneDrive\Desktop\trun\data analytic and deep learn\iris.csv")

plt.figure(figsize=(8, 6))
ax = plt.gca() 

for species in iris_df['species'].unique():
    species_df = iris_df[iris_df['species'] == species]
    ax.scatter(species_df['sepal_length'], species_df['sepal_width'], label=species, alpha=0.7) #alpha để độ trong suốt

ax.set_title('Biểu đồ phân tán giữa Sepal Length và Sepal Width (phân biệt theo loài)')
ax.set_xlabel('Sepal Length (cm)')
ax.set_ylabel('Sepal Width (cm)')
ax.legend(title='Loài hoa')

ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.xaxis.set_ticks_position('bottom')
ax.yaxis.set_ticks_position('left')

plt.grid(True, linestyle='--', alpha=0.5)
plt.show()

plt.figure(figsize=(8,6))
sns.scatterplot(x='sepal_length', y='sepal_width', hue='species', data=iris_df)
plt.title('Biểu đồ phân tán giữa Sepal Length và Sepal Width (phân biệt theo loài) - dùng Seaborn')
plt.show()