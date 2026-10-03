import numpy as np
#11
massive = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11])
massive.reshape(3, 4)
#12
print(massive.shape, massive.ndim, massive.size, massive.dtype)
#13
massive.flatten()
#14
matrix_2x3 = np.array([[1, 2, 3], [4, 5, 6]])
matrix_3x2 = matrix_2x3.T
#15
massive1 = np.array([1, 2, 3])
massive2 = np.array([4, 5, 6])
resultMass = np.vstack((massive1, massive2))
resultMass2 = np.hstack((massive1, massive2))
#16
massive9 = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9])
np.split(massive9, 3)
#17
massive5 = np.array([1, 2, 3, 4, 5])
massive5.reshape(5, 1)
#18
massive_strange = np.array([1, 5, 1, 3])
np.squeeze(massive_strange)
#19
array1 = np.array([[1, 2, 3], [4, 5, 6]])
array2 = np.array([[7, 8, 9], [10, 11, 12]])
result_stack = np.stack((array1, array2), axis=0)
#20
array_3d = np.ones((2, 3, 4))
swapped_array = np.swapaxes(array_3d, 0, 1)
