my_dict = {'a': 1, 'b': 2, 'c': 3, 'd': 4, 'e': 5}
items = list(my_dict.items())
items[0], items[-1] = items[-1], items[0]
del items[1]
my_dict.clear()
my_dict.update(items)
my_dict['ney_key'] = "ney_value"
print(my_dict)