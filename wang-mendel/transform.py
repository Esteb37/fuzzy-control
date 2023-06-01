import numpy as np

data = np.load(
    "C:/Users/esteb/OneDrive/Documents/Escuela/Robótica/Fuzzy/wang-mendel/datos.npy")

text = ""
for i, row in enumerate(data, 1):
    text += f"\n\"{i}\" {row[0]} {row[1]} {row[2]}"
