class Nikola:
  __slots__ = ('name', 'age')

  def __init__(self, name, age):
    self.age = age
    if name == 'Николай':
      self.name = 'Николай'
    else:
      self.name = f'Я не {name}, а Николай'


user1 = Nikola('Максим', 25)
print(user1.name)
print(user1.age)

user2 = Nikola('Николай', 30)
print(user2.name)

user1.age = 26
print(user1.age)
