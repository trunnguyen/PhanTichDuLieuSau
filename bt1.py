import numpy as np

array_1 = np.arange(1, 21)
print("Mảng từ 1 đến 20:", array_1)

print("Tổng:", np.sum(array_1))
print("Giá trị lớn nhất:", np.max(array_1))
print("Giá trị nhỏ nhất:", np.min(array_1))
print("Trung bình:", np.mean(array_1))

array_2d = np.random.randint(0, 101, size=(3, 5))
print("Mảng 2D:", array_2d)
print("Hàng thứ 2:", array_2d[1, :])  
print("Cột thứ 3:", array_2d[:, 2])  

print("Phần tử ở hàng thứ 2, cột thứ 3:", array_2d[1,2])