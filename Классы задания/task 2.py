class TriangleChecker:

    def __init__(self, sides):
        self.sides = sides

    def is_triangle(self):
        for side in self.sides:
            if type(side) not in (int, float):
                return "Нужно вводить только числа!"

        if any(side <= 0 for side in self.sides):
            return "С отрицательными числами ничего не выйдет!"

        a, b, c = self.sides

        if a + b > c and a + c > b and b + c > a:
            return "Ура, можно построить треугольник!"
        else:
            return "Жаль, но из этого треугольник не сделать."

checker1 = TriangleChecker([3, 4, 5.5])
print(checker1.is_triangle())

checker2 = TriangleChecker([3, 4, "5"])
print(checker2.is_triangle())

checker3 = TriangleChecker([2, 3, 10])
print(checker3.is_triangle())
