class Soda:
    def __init__(self, addictive = None):
        self.addictive = addictive

    def showMyDrink(self):
        if(self.addictive):
            print("Газировка и " +str(self.addictive))
        else:
            print("Обычная газировка")

drink1 = Soda("Яблоко")
drink1.showMyDrink()

drink2 = Soda()
drink2.showMyDrink()