import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

x = np.linspace(-10, 10, 400)  
y = x**2

plt.figure(figsize=(8, 6))  
plt.plot(x, y)
plt.title('Đồ thị hàm số y = x^2')
plt.xlabel('x')
plt.ylabel('y')
plt.grid(True)  
plt.show()

data = {'Name': ['Alice', 'Bob', 'Charlie', 'David', 'Eva'],
        'Age': [23, 25, 22, 24, 21],
        'Score': [85, 90, 78, 92, 88]}
df = pd.DataFrame(data)

plt.figure(figsize=(8, 6))
plt.bar(df['Name'], df['Score'])
plt.title('Biểu đồ điểm số của sinh viên')
plt.xlabel('Tên sinh viên')
plt.ylabel('Điểm số')
plt.ylim(0, 100) 
plt.show()

try:
    iris_df = pd.read_csv("Iris.csv")
    iris_df.rename(columns={'Species': 'species'}, inplace=True) #Đổi tên cột cho ngắn gọn
except FileNotFoundError:
    print("Lỗi: Không tìm thấy file Iris.csv. Hãy chắc chắn rằng file đã được tải xuống và nằm trong cùng thư mục với script Python hoặc cung cấp đường dẫn đầy đủ.")
    exit()

species_counts = iris_df['species'].value_counts()

plt.figure(figsize=(8, 6))
plt.pie(species_counts, labels=species_counts.index, autopct='%1.1f%%', startangle=90)
plt.title('Phần trăm mỗi loại hoa Iris')
plt.axis('equal')  
plt.show()