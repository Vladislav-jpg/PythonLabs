import numpy as np
# 80
arr_80 = np.random.uniform(-5, 5, 20)
np.sort(arr_80)
# 81
arr1_81, arr2_81 = np.array([1, 2, 3]), np.array([1, 2, 3])
np.array_equal(arr1_81, arr2_81)
# 82
np.isnan([1, np.nan, 3, np.nan, 5]).nonzero()[0]
# 83
arr_83 = np.array([1.0, np.nan, 3.0, 4.0])
arr_83[np.isnan(arr_83)] = np.nanmean(arr_83)
# 84
arr_84 = np.array([1.0, np.inf, 3.0, -np.inf])
arr_84[np.isinf(arr_84)] = 0
# 85
arr_85 = np.array([10, 50, 150])
print(np.all(arr_85 > 0), np.any(arr_85 > 100))
# 86
arr1_86, arr2_86 = np.array([1.000001, 2.0]), np.array([1.000000, 2.0])
np.allclose(arr1_86, arr2_86)
# 87
arr1_87, arr2_87 = np.array([True, False, True]), np.array([False, False, True])
np.logical_and(arr1_87, arr2_87)
np.logical_or(arr1_87, arr2_87)
# 88
np.intersect1d([1, 2, 3, 4, 5], [3, 4, 5, 6, 7])
# 89
arr1_89, arr2_89 = np.array([1, 2, 3, 4]), np.array([3, 4, 5, 6])
np.setdiff1d(arr1_89, arr2_89)
# 90
arr_90 = np.array([3.0, np.nan, 1.0, 2.0, np.nan])
np.sort(arr_90)