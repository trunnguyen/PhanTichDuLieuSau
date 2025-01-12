import pandas as pd

data = {'Name': ['Alice', 'Bob', 'Charlie', 'David', 'Eva'],
        'Age': [23, 25, 22, 24, 21],
        'Score': [85, 90, 78, 92, 88]}
df = pd.DataFrame(data)
print("DataFrame:\n", df)

mean_score = df['Score'].mean()
print("\nGiá trị trung bình của cột 'Score':", mean_score)

filtered_df = df[df['Score'] > 85]
print("\nCác hàng có Score > 85:\n", filtered_df)

filtered_df_query = df.query('Score > 85')
print("\nCác hàng có Score > 85 :\n",filtered_df_query)