import numpy as np
filedata = np.genfromtxt('data.txt', delimiter=',')
filedata = filedata.astype('int32')

print(filedata > 5)
print(filedata[filedata > 3])

filedata[filedata < 0] = 0
print(filedata)

print((filedata > 5).sum())