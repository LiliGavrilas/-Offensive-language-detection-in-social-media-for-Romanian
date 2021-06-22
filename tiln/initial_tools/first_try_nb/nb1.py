import csv
import pandas as pd
import numpy as np
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from string import punctuation
import snowballstemmer
from nltk.stem.wordnet import WordNetLemmatizer


def tokenize(examplu_propozitie_ro):
    propozitie_tokenizata = word_tokenize(examplu_propozitie_ro)  # tokenizarea merge in romana

    stop_words_ro = set(stopwords.words('romanian'))
    propozitie_tokenizata_fara_stopwords = [w for w in propozitie_tokenizata if
                                            not w in stop_words_ro]  # scoaterea de stopwords merge si in romana

    propozitie_tokenizata_fara_stopwords_si_puncte = [w for w in propozitie_tokenizata_fara_stopwords if
                                                      not w in punctuation]

    return propozitie_tokenizata_fara_stopwords_si_puncte

vocabulary = {}
data = pd.read_csv('allcomm.csv')
'''
with open('allcomm.csv', mode='a+', newline='', encoding='utf-8') as employee_file:
    employee_writer = csv.writer(employee_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
    with open('offensive.txt') as fp:
       line = fp.readline()
       while line:
           x = (line.strip().split(',', 1))
           employee_writer.writerow([x[0], x[1]])
           line = fp.readline()
'''


def build_vocabulary(curr_comm):
    idx = len(vocabulary)
    
    for word in curr_comm:
        if word.lower() not in vocabulary:
            vocabulary[word] = idx
            idx += 1
            
if __name__ == '__main__':
    for i in range(data.shape[0]):
        curr_comm = tokenize(str(data.iloc[i,1]))
        print(f"Current comment is {i}/{data.shape[0]} and the \
                length of vocabulary is {len(vocabulary)}")

        build_vocabulary(curr_comm)

    file = open("vocabulary.txt", "w")
    file.write(str(vocabulary))
    file.close()




