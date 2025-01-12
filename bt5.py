import pandas as pd
import numpy as np 

data_with_missing = {'Name': ['Alice', 'Bob', 'Charlie', 'David', 'Eva'],
                     'Age': [23, np.nan, 25, 24, 22],
                     'City': ['New York', 'Boston', np.nan, 'Chicago', 'Boston'],
                     'Salary': [60000, 52000, np.nan, 58000, np.nan]}

df_missing = pd.DataFrame(data_with_missing)
print("Dữ liệu ban đầu:\n", df_missing)

df_missing['Age'].fillna(df_missing['Age'].mean(), inplace=True)
print("\nSau khi điền giá trị thiếu ở cột Age:\n", df_missing)

df_missing_dropped = df_missing.dropna(thresh=3) 
print("\nSau khi xóa các hàng có nhiều hơn 1 giá trị thiếu:\n", df_missing_dropped)

df_missing_dropped['Salary'].fillna(50000, inplace=True)
print("\nSau khi điền giá trị thiếu ở cột Salary bằng 50000:\n", df_missing_dropped)