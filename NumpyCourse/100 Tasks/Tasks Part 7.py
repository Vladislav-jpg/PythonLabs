import numpy as np
#61
np.dot([1,2,3], [4,5,6])
#62
np.random.rand(2, 3) @ np.random.rand(3, 2)
#63
# Определитель существует только для квадратных матриц.
# Пример для матрицы 2x2:
np.linalg.det(np.random.rand(2, 2))
#64
# Обратная матрица существует только для квадратных матриц.
# Пример для 3x3:
A = np.random.rand(3,3)
np.allclose(A @ np.linalg.inv(A), np.eye(3))
#65
np.linalg.solve(np.random.rand(3, 3), np.random.rand(3))
#66
np.linalg.eig(np.random.rand(3, 3))
#67
np.linalg.matrix_rank(np.random.rand(4, 3))
#68
np.linalg.norm([3, 4])
#69
A = np.random.rand(3, 3)
U, S, V = np.linalg.svd(A)
# Восстановление:
U @ np.diag(S) @ V
#70
A = np.random.rand(4, 2)
A.T @ A