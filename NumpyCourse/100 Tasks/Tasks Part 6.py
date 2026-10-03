import numpy as np
#51
np.zeros((3, 3)) + np.array([1, 2, 3])
#52
np.zeros((3, 3)) + np.array([1, 2, 3]).reshape(3, 1)
#53
a = np.arange(1, 11)
a[:, np.newaxis] * a
#54
arr = np.random.rand(4, 3)
arr / np.sum(arr, axis=1, keepdims=True)
#55
a, b = np.arange(5), np.arange(5)
a[:, np.newaxis] - b
#56
arr = np.random.rand(5, 3)
arr - np.mean(arr, axis=0)
#57
arr = np.random.rand(3, 4)
w = np.array([1, 2, 3])
arr * w[:, np.newaxis]
#58
# (3,1) и (1,4) - совместимы
# (3,4) и (4,) - совместимы
# (2,3) и (3,2) - несовместимы
#59
pts = np.random.rand(5, 2)
np.sqrt(np.sum(pts**2, axis=1))
#60
arr = np.random.rand(4, 3)
(arr - np.mean(arr, axis=0)) / np.std(arr, axis=0)