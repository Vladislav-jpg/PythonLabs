class KgToPounds:
    def __init__(self, kg):
        self.__kg = kg

    def to_pounds(self):
        return self.__kg * 2.205

    def _vzyat_kg(self):
        return self.__kg

    def _pomenyat_kg(self, new_kg):
        if isinstance(new_kg, (int, float)):
            self.__kg = new_kg
        else:
            raise ValueError('Килограммы задаются только числами')

    kg = property(_vzyat_kg, _pomenyat_kg)

box = KgToPounds(10)


print("Вес в кг:", box.kg)
print("Вес в фунтах:", box.to_pounds())

box.kg = 25
print("Новый вес в кг:", box.kg)
print("Новый вес в фунтах:", box.to_pounds())

try:
    box.kg = "двадцать килограмм"
except ValueError as e:
    print("Except")
