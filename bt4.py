import pandas as pd

iris_df = pd.read_csv(r"C:\Users\nguye\OneDrive\Desktop\trun\data analytic and deep learn\iris.csv") 


print("Thông tin tổng quan về dữ liệu:",iris_df.info())
print("\nMô tả dữ liệu:",iris_df.describe())

print("\nTrung bình sepal_length:", iris_df['sepal_length'].mean())
print("Giá trị lớn nhất sepal_length:", iris_df['sepal_length'].max())
print("Giá trị nhỏ nhất sepal_length:", iris_df['sepal_length'].min())