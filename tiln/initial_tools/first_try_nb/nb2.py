# import pandas as pd
# import numpy as np
# import ast
#
# data = pd.read_csv('allcomm.csv')
# file = open('vocabulary.txt', 'r')
# contents = file.read()
# vocabulary = ast.literal_eval(contents)
#
# X = np.zeros((data.shape[0], len(vocabulary)))
# y = np.zeros((data.shape[0]))
#
# for i in range(data.shape[0]):
#     comm = str(data.iloc[i,1]).split()
#
#     for comm_word in comm:
#         if comm_word.lower() in vocabulary:
#             X[i, vocabulary[comm_word]] += 1
#             y[i] = data.iloc[i,0]
#
# np.save('X.npy', X)
# np.save('y.npy', y)
import csv

with open('corpus_final.csv', mode='a+', newline='', encoding='utf-8') as employee_file:
    employee_writer = csv.writer(employee_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
    with open('C:/Users/Florentin/Desktop/TILN/test/GetManhuaProject/nonoffensive.txt') as fp:
       line = fp.readline()
       while line:
           x = (line.strip().split(',', 1))
           employee_writer.writerow([x[0], x[1]])
           line = fp.readline()
