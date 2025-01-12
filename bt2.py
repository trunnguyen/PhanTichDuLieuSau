import numpy as np

random_array = np.random.rand(20)
print("Mảng ngẫu nhiên từ 0 đến 1:", random_array)

min_val = np.min(random_array)
max_val = np.max(random_array)
normalized_array = (random_array - min_val) / (max_val - min_val)

print("Mảng sau khi chuẩn hóa (trong trường hợp này không thay đổi):", normalized_array)

a = np.array([1, 2, 3])
b = np.array([4, 5, 6])
dot_product = np.dot(a, b)
print("Tích vô hướng của a và b:", dot_product)


matrix = np.random.rand(5, 5)
print("Ma trận:\n", matrix)
determinant = np.linalg.det(matrix)
print("Định thức của ma trận:", determinant)
if determinant != 0:
    inverse_matrix = np.linalg.inv(matrix)
    print("Ma trận nghịch đảo:\n", inverse_matrix)
else:
    print("Ma trận không khả nghịch (định thức = 0).")