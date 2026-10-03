import numpy as np
#71
np.random.seed(42)
np.random.randint(1, 101, 5)
#72
np.random.normal(50, 10, 1000)
#73
arr = np.arange(10)
np.random.shuffle(arr)
#74
np.random.choice([10,20,30,40,50], 3, replace=False)
#75
arr = np.random.randint(0, 10, (3, 3))
print(np.max(arr), np.unravel_index(np.argmax(arr), arr.shape))
#76
coins = np.random.randint(0, 2, 1000)
np.mean(coins == 1)
#77
np.sum(np.random.rand(100) > 0.5)
#78
np.random.permutation(10)
#79
dice = np.random.randint(1, 7, 6000)
np.unique(dice, return_counts=True)
#80
np.sort(np.random.uniform(-5, 5, 20))