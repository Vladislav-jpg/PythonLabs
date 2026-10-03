import numpy as np
#41
arr = np.random.rand(100)
print(np.sum(arr), np.mean(arr), np.min(arr), np.max(arr))
#42
arr = np.random.rand(4, 5)
print(np.sum(arr, axis=1), np.sum(arr, axis=0))
#43
a = np.array([2, 4, 4, 4, 5, 5, 7, 9])
print(np.std(a), np.var(a))
#44
arr = np.random.rand(10)
print(np.argmax(arr), np.argmin(arr))
#45
arr = np.random.rand(100)
print(np.median(arr), np.percentile(arr, 25), np.percentile(arr, 75))
#46
a = np.array([5, 2, 9, 1, 7])
print(np.sort(a), np.argsort(a))
#47
a = np.array([1,2,3,4,5])
print(np.cumsum(a), np.cumprod(a))
#48
arr = np.random.randint(0, 10, 100)
np.bincount(arr)
#49
np.corrcoef(np.random.rand(10), np.random.rand(10))
#50
np.unique([1,2,2,3,3,3,4], return_counts=True)