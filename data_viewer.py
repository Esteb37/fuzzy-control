import numpy as np
import pandas as pd

data = np.load("results.npy")


antecedents = [3, 5, 7, 9, 11, 13, 15, 17, 19]

dic = {}

for i in antecedents:
    for j in antecedents:
        for k in antecedents:
            num_id = int("".join(map(str, (i, j, k))))
            if num_id in data[:, 0]:
                dic[num_id] = data[np.where(data[:, 0] == num_id)[0][0], 1:]

print(len(dic))
