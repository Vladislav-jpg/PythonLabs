import numpy as np
# 91
grades = np.random.randint(2, 6, (5, 4)) # 5 студентов, 4 предмета
print("Среднее по студентам:", np.mean(grades, axis=1))
print("Среднее по предметам:", np.mean(grades, axis=0))
# 92
img = np.random.randint(0, 256, (10, 10))
img_normalized = img / 255.0
# 93
ts = np.random.rand(20)
moving_avg = np.convolve(ts, np.ones(3)/3, mode='valid')
# 94
labels = np.array([0, 2, 1, 0, 3])
one_hot = np.eye(np.max(labels) + 1)[labels]
# 95
arr_95 = np.random.randn(10)
k = 3
closest_to_zero = arr_95[np.argsort(np.abs(arr_95))[:k]]
# 96
pts1, pts2 = np.random.rand(5, 2), np.random.rand(3, 2)
distances = np.sqrt(np.sum((pts1[:, np.newaxis] - pts2)**2, axis=-1))
# 97
board = np.zeros((5, 5), dtype=int)
board[1::2, ::2] = 1
board[::2, 1::2] = 1
# 98
sig = np.random.rand(10)
kernel = np.array([1, 0.5, 0.2])
conv_result = np.array([np.sum(sig[i:i+3] * kernel) for i in range(len(sig)-2)])
# 99
arr_99 = np.random.rand(4, 5)
idx = np.argmax(np.sum(arr_99, axis=1))
print(idx, arr_99[idx])
# 100
arr_100 = np.random.rand(3, 4)
e_x = np.exp(arr_100 - np.max(arr_100, axis=1, keepdims=True))
softmax = e_x / np.sum(e_x, axis=1, keepdims=True)