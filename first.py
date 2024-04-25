import math
import random
import converter
import ecomerce.shipping
#SAYILARI YAZDIRMA
"""" 
number = input("Phone:")
numbers = {
    '0': "Zero",
    "1": "One",
    "2": "Two",
    "3": "Three",
    "4": "Four",
    "5": "Five",
    "6": "Six",
    "7": "Seven",
    "8": "Eight",
    "9": "Nine"
}
i = 0
numberr = ""
for ch in number:
    numberr += numbers.get(ch) + " "
print(numberr)
"""
class allOf:
    def sing(self):
        print("yakup and melo")

class Musician(allOf):
    def __init__(self,x,y):
        self.x = x
        self.y = y
    def move(self):
        print("move")

musician = Musician("Yakup","Kavak")
musician.sing()

converter.yazdir()
ecomerce.shipping.calc_ship()