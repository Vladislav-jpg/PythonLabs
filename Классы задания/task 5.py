from functools import total_ordering

@total_ordering
class RealString:
    def __init__(self, some_str):
        self.some_str = str(some_str)

    def __eq__(self, other):
        if not isinstance(other, RealString):
            other = RealString(other)
        return len(self.some_str) == len(other.some_str)

    def __gt__(self, other):
        if not isinstance(other, RealString):
            other = RealString(other)
        return len(self.some_str) > len(other.some_str)

str1 = RealString("Apple")
str2 = RealString("Яблоко")

print(str1 < str2)
print(str1 > str2)

print(str1 == "Banana")
print(str1 < "Апельсин")
print("Груша" < str1)
