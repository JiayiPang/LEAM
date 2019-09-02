import numpy as np
import pandas as pd
import random
lista = [1,2,3,4,5,6,7,8,9]
slice = random.sample(lista, 5)
while(lista):
    choice = random.choice(lista)
    lista.remove(choice)
    print(lista)