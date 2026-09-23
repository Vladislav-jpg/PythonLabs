import numpy as np
a = np.array([1, 2, 3])
b = a
b[0] = 5
print(a)
print(b)
'''ОБЫЧНОЕ ПРИСВАИВАНИЕ b = a ПРОСТО СОЗДАЕТ ССЫЛКУ НА ПЕРВЫЙ ЭЛЕМЕНТ В ПАМЯТИ, ПОЭТОМУ МЕНЯЮТСЯ ОБА'''

a = np.array([1, 2, 3])
b = a.copy()
b[0] = 5
print(a)
print(b)



v = np.arange(12)
print(v.reshape((3, 4)))
print(v.reshape((4, 3)))
print(v.reshape((2, 2, 3)))
# print(v.reshape((5, 3))) # Вызовет ошибку
'''ОШИБКА reshape ПОТОМУ ЧТО НЕЛЬЗЯ 12 ЭЛЕМЕНТОВ ПЕРЕДЕЛАТЬ В ФОРМУ НА 15 ЭЛЕМЕНТОВ RESIZE МОЖЕТ ИЗМЕНЯТЬ РАЗМЕР ОТБРАСЫВАЯ ИЛИ ПОВТОРЯЯ ЭЛЕМЕНТЫ'''





v1 = np.array([1, 2, 3])
v2 = np.array([9, 8, 7])
print(np.vstack([v1, v2]))

m1 = np.ones((2, 3))
m2 = np.zeros((2, 3))
print(np.hstack((m1, m2)))

mat1 = np.ones((2, 2))
mat2 = np.zeros((2, 2))
print(np.stack((mat1, mat2), axis=0))
print(np.stack((mat1, mat2), axis=1))
'''stack ДОБАВЛЯЕТ НОВУЮ ОСЬ, А ПРЕДЫДУЩИЕ ОСИ УВЕЛИЧИВАЮТ СВОИ ИНДЕКСЫ НА ЕДИНИЦУ ЧТОБЫ ОСВОБОДИТЬ МЕСТО vstack И hstack ОБЪЕДИНЯЮТ ВДОЛЬ УЖЕ СУЩЕСТВУЮЩИХ ОСЕЙ'''






a = np.array([[1, 2, 3], [4, 5, 6]])
print(a.transpose())
print(a.T)
print(a.shape, a.transpose().shape)

a3d = np.zeros((1, 2, 3))
print(a3d.transpose(1, 2, 0).shape)
'''ПЕРВАЯ ОСЬ СТАЛА ТРЕТЬЕЙ, ВТОРАЯ СТАЛА ПЕРВОЙ А ТРЕТЬЯ СТАЛА ВТОРОЙ ФОРМА СТАЛА (2, 3, 1)'''