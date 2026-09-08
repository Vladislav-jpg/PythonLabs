def divider(a, b):
    return (b == 0) and "Нули в знаменателе не приветствуются" or (a/b)**3

print(divider(2, 9))
print(divider(3, 0))