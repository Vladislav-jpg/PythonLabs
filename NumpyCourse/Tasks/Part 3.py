import numpy as np

m_zeros = np.zeros((3, 4), dtype='int32')
m_ones = np.ones((2, 3, 3))
m_full = np.full((4, 4), -7)
m_eye = np.eye(5)

print("Zeros dtype: ", m_zeros.dtype)
print("Ones dtype: ",m_ones.dtype)
print("Full dtype: ", m_full.dtype)
print("Eye dtype: ", m_eye.dtype)

# По умолчанию для функций zeros ones и eye элементам присваивается тип float64

rand_ints = np.random.randint(-10, 11, size=(5, 5))
rand_floats = np.random.rand(5, 5)
print("rand_ints: ",rand_ints)
print("rand_floats: ",rand_floats)
# Функция np.zeros принимает размер в виде кортежа поскольку у нее есть дополнительный параметр dtype если передать без скобок np.zeros(2, 2) то пайтон подумает вторую двойку за указание типа данных
# Функция np.random.rand не имеет параметра dtype поэтому размерности передаются простым перечислением аргументов через запятую