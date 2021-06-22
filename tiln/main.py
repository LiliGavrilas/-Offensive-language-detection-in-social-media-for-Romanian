import nltk

nltk.download('punkt')
nltk.download('stopwords')
import json
import requests
import re
import datetime
import pickle
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import VotingClassifier
from sklearn import svm

from nltk.corpus import stopwords
from stop_words import get_stop_words
from nltk.tokenize import word_tokenize
from string import punctuation


# Functii :
def lematizare_si_pos_tag_racai(comment):
    # Daca trimitem comment-ul ca o lista il refacem in comment cu spatiu intre cuvinte pentru a nu face prea multe request-uri la API
    # TO DO?:Probabil chiar sa trimitem mai multe comment-uri odata
    if isinstance(comment, list):
        comment_string = ""
        for cuvant in comment:
            comment_string = comment_string + " " + cuvant
        comment = comment_string

    racai_url = 'http://relate.racai.ro:5000/process'
    comment_data = {'text': comment,
                    'exec': 'lemmatization'}

    racai_answer = requests.post(racai_url, data=comment_data)
    racai_answer_json = json.loads(racai_answer.text)
    lemmas = []
    # Procesare answer :
    racai_answer_json_tokens = racai_answer_json["teprolin-result"]["tokenized"][0]
    for token in racai_answer_json_tokens:
        # print(token["_lemma"], token["_ctg"], token["_deprel"], token["_msd"])
        lemmas.append((token["_lemma"], token["_ctg"]))

    return lemmas


def read_corpus_file(corpus_file_path):
    """ Functie de citire csv corpus si spargere in vectori de categorii cu toate commenturile din aceiasi categorie"""
    """ Csv de forma asta pe linie: "categorie,comment" """
    """ categorie=0 ->offensive categorie=1 ->neutru categorie=2 ->nonoffensive """

    corpus = open(corpus_file_path, 'r', encoding='utf-8')

    linie_gresita = 0

    corpus_offensive = []
    corpus_nonoffensive = []
    corpus_neutru = []

    for linie in corpus:
        # scoatem enter
        linie = linie.strip()
        # spargem doar dupa prima virgula
        linie_sparta = linie.split(",", 1)
        if len(linie_sparta) == 2:
            categorie, comment = linie_sparta
            if (categorie == "1"):
                corpus_offensive.append(comment)
            # elif (categorie == "2"):
            #     corpus_neutru.append(comment)
            elif (categorie == "0"):
                corpus_nonoffensive.append(comment)
            else:
                # print("Linie gresita tip1 : categorie=" , categorie , "comment=",comment)
                linie_gresita = linie_gresita + 1
        else:
            # print("Linie gresita tip2:" , linie)
            linie_gresita = linie_gresita + 1
        total = len(corpus_offensive) + len(corpus_nonoffensive) + len(corpus_neutru) + linie_gresita

    # print("Avem" , total , " linii citite corect si " , linie_gresita , " incorecte in fisier ")
    print("Avem", total, " linii citite corect")
    # print(" Offensive =" , len(corpus_offensive) , "\n Nonoffensive =" , len(corpus_nonoffensive) , "\n Neutru =" , len(corpus_neutru))
    print(" Offensive =", len(corpus_offensive), "\n Nonoffensive =", len(corpus_nonoffensive))
    return corpus_offensive, corpus_nonoffensive, corpus_neutru


def filtrare_comment(initial_comment):
    # Filtre substitutie greseli/abrevieri:
    comment_filtered = re.sub(r'(^|\s)pt(\s|$)', ' pentru ', initial_comment, re.IGNORECASE, re.UNICODE)
    comment_filtered = re.sub(r'(^|\s)ditre(\s|$)', ' dintre ', initial_comment, re.IGNORECASE, re.UNICODE)
    comment_filtered = re.sub(r'(^|\s)lassa(\s|$)', ' lasa ', initial_comment, re.IGNORECASE, re.UNICODE)
    comment_filtered = re.sub(r'(^|\s)c..r(\s|$)', ' cur ', comment_filtered, re.IGNORECASE, re.UNICODE)
    comment_filtered = re.sub(r'(^|\s)c..ve(\s|$)', ' curve ', comment_filtered, re.IGNORECASE, re.UNICODE)
    comment_filtered = re.sub(r'ull*(\s|$)', 'ul ', comment_filtered, re.IGNORECASE, re.UNICODE)
    comment_filtered = re.sub(r'iii*(\s|$)', 'ii ', comment_filtered, re.IGNORECASE, re.UNICODE)

    comment_filtered = re.sub(r'..', ' ', comment_filtered, re.IGNORECASE, re.UNICODE)
    # end
    if 0:
        f = open("test_filtrare.txt", "a")
        f.write("Initial: " + initial_comment)
        f.write("Filtrat: " + comment_filtered)
        f.close()
    return comment_filtered


def procesare_comments(initial_comments, prefiltrare=0):
    comments_processed = []
    for num, comment in enumerate(initial_comments, start=1):
        # if (num <= 10):# lucrez temporar doar pe o parte din comment-uri
        # examplu_comment_ro = """Aceasta este un comment, pe care incerc sa il tokenizez, sa ii scot stopwords , semnele de punctuatie si sa il lematizez."""
        comment_intial = comment

        # 1.1.0.0 prefiltrare optionala:
        if prefiltrare == 1:
            comment = filtrare_comment(comment)

        # 1.1.0 lowercase si alte cazuri speciale de indepartat
        comment = comment.lower()
        comment = re.sub(r'[^\w\s]', ' ', comment, re.UNICODE)
        comment = re.sub(r'[\s\d*\s]', ' ', comment, re.UNICODE)

        # 1.1.1 tokenizare
        comment_tokenizat = word_tokenize(comment)
        # print("Comment ro tokenizat:\n",comment_tokenizat)

        # 1.1.2 stergem stopwords si cuvintele de 1 caracter
        stop_words_ro = set(stopwords.words('romanian'))
        # comment_fara_stopwords = [w for w in comment_tokenizat if ((not w in stop_words_ro) and  len(w) >= 2) ]
        comment_fara_stopwords = [w for w in comment_tokenizat if (not w in stop_words_ro)]

        # print("Comment ro tokenizat fara stopwords:\n",comment_fara_stopwords)

        # 1.1.3 stergem punctuatia
        comment_fara_puncte = [w for w in comment_fara_stopwords if (not w in punctuation)]
        # print("Comment ro tokenizat fara stopwords si puncte:\n",comment_fara_puncte)
        if 0:
            f = open("test_filtrare.txt", "a")
            f.write("Procesat: " + str(comment_fara_puncte))
            f.close()
        # 1.1.4 transformare in lema si pos tag -> dureaza prea mult ->trebuie optimizata cumva
        toate_cuvintele_comment = []
        """
        comment_lemmizat_si_postag = lematizare_si_pos_tag_racai(comment_fara_puncte)#pentru fiecare comment extragem lema si pos tag-ul pe baza API racai
        # print("Comment ro tokenizat fara stopwords si puncte + lematizare si pos tagging:\n",comment_lemmizat_si_postag)
        for no_cuv ,tupla_lema_tag in enumerate(comment_lemmizat_si_postag, start=0):
          lema = tupla_lema_tag[0]
          tag = tupla_lema_tag[1]
          # ('urechi', 'NOUN')
          # ('vorbi', 'VERB')
          # ('băiat', 'NOUN')
          # ('alin', 'ADV')
          # ('tu', 'PRON')
          # ('avea', 'AUX')
          if(re.search(r"\s|\d",lema) or len(lema) <=1 ):
            print("\n\t\t!!! Posibila erroare prepocesare ce ar trebui indepartata in lema:", lema,tag,comment_fara_puncte[no_cuv] , comment_intial)
          toate_cuvintele_comment.append(lema)#deocamdata nu folosim pos tag-ul
        """
        # Reconstruim commentul din cuvintele procesate separate de spatii
        comment_processed = ""
        for cuvant in comment_fara_puncte:
            comment_processed = comment_processed + " " + cuvant
        comments_processed.append(comment_processed)

    return comments_processed


def print_results(model_name, predicted_labels, initial_labels, duration_train, duration_test):
    tn, fp, fn, tp = confusion_matrix(predicted_labels, initial_labels).ravel()
    o_precision = round((tn / (tn + fn)) * 100, 3)
    o_recall = round((tn / (tn + fp)) * 100, 3)
    no_precision = round((tp / (tp + fp)) * 100, 3)
    no_recall = round((tp / (tp + fn)) * 100, 3)
    o_f1_score = 2 * (o_precision * o_recall) / (o_precision + o_recall)
    no_f1_score = 2 * (no_precision * no_recall) / (no_precision + no_recall)
    print(model_name + " Train duration:", duration_train.total_seconds(), " sec", " Test duration:",
          duration_test.total_seconds(), " sec")
    print(model_name + " Offensive - TN:", tn, "FN:", fn, "Precision:", o_precision, "%", "Recall:", o_recall, "%",
          "F1 score:", o_f1_score)
    print(model_name + " NonOffensive - TP:", tp, "FP:", fp, "Precision:", no_precision, "%", "Recall:", no_recall, "%",
          "F1 score:", no_f1_score)


# MAIN:

# antrenare si testare principala
prefiltrare = 0

if 1:
    # 0. citire comments:
    offensive, nonoffensive, neutru = read_corpus_file('./content/corpus12mii_final.csv')  # var 1
    # offensive , nonoffensive , neutru = read_corpus_file('/content/copus_lemizat_var2.csv') #var 2
    # offensive , nonoffensive , neutru = read_corpus_file('/content/copus_lemizat_cu_posstag_var2.csv') #var 3
    # print("Corpus offensinve" , offensive)
    # print("Corpus nonoffensive" , nonoffensive)
    # print("Corpus neutru" , neutru)

    # 1 procesare comments and add labels:
    all_comments = np.concatenate((offensive, nonoffensive))
    # all_comments = np.concatenate((all_comments, neutru))
    all_lables = [1] * len(offensive) + [0] * len(nonoffensive)
    print(len(all_comments), len(all_lables))
    all_comments_processed = procesare_comments(all_comments, prefiltrare)
    # all_comments_processed = procesare_comments(all_comments)
    stop_words_ro = get_stop_words('ro')

if 1:
    # Impartire comments/labels in date antrenare si date test
    train_comments, test_comments, train_labels, test_labels = train_test_split(all_comments_processed, all_lables,
                                                                                test_size=0.2, random_state=0)

    comments_word_conts = CountVectorizer(stop_words=stop_words_ro)
    train_data = comments_word_conts.fit_transform(train_comments)
    tfidf_transformer = TfidfTransformer(sublinear_tf=True)
    train_tfidf = tfidf_transformer.fit_transform(train_data)

    pickle.dump(comments_word_conts.vocabulary_, open("feature.pkl", "wb"))

    test_data = comments_word_conts.transform(test_comments)
    test_tfidf = tfidf_transformer.transform(test_data)

    # 2 antrenare MultinomialNB
    start_time = datetime.datetime.now()
    multi_naive_bayes = MultinomialNB()
    multi_naive_bayes.fit(train_tfidf, train_labels)
    end_train_time = datetime.datetime.now()
    # 3 testare MultinomialNB
    test_predicted_nb = multi_naive_bayes.predict(test_tfidf)
    end_test_time = datetime.datetime.now()
    # 4 Afisare rezultate separate pe categorii(False/True positiv, False/True negativ)
    # confusion matrix
    print_results("Multinomial Naive Bayes", test_predicted_nb, test_labels, end_train_time - start_time,
                  end_test_time - end_train_time)
    pickle.dump(multi_naive_bayes, open("multi_naive_bayes_model.pickle", "wb"))

    # 2 antrenare Passive Aggressive
    start_time = datetime.datetime.now()
    pa = PassiveAggressiveClassifier(C=0.5, random_state=5)
    pa.fit(train_tfidf, train_labels)
    end_train_time = datetime.datetime.now()
    # 3 testare Passive Aggressive
    test_predicted_pa = pa.predict(test_tfidf)
    end_test_time = datetime.datetime.now()
    # 4 Afisare rezultate separate pe categorii(False/True positiv, False/True negativ)
    print_results("Passive Aggressive", test_predicted_pa, test_labels, end_train_time - start_time,
                  end_test_time - end_train_time)
    pickle.dump(pa, open("pa_model.pickle", "wb"))

    # 2 antrenare SVM
    start_time = datetime.datetime.now()
    svm = svm.SVC(C=1000)
    svm.fit(train_tfidf, train_labels)
    end_train_time = datetime.datetime.now()
    # 3 testare SVM
    test_predicted_svm = svm.predict(test_tfidf)
    end_test_time = datetime.datetime.now()
    # 4 Afisare rezultate separate pe categorii(False/True positiv, False/True negativ)
    print_results("SVM", test_predicted_svm, test_labels, end_train_time - start_time, end_test_time - end_train_time)
    pickle.dump(svm, open("svm_model.pickle", "wb"))

    # 2 antrenare Logistic Regression
    start_time = datetime.datetime.now()
    lr = LogisticRegression(solver='liblinear', random_state=0)
    lr.fit(train_tfidf, train_labels)
    end_train_time = datetime.datetime.now()
    # 3 testare Logistic Regression
    test_predicted_lr = lr.predict(test_tfidf)
    end_test_time = datetime.datetime.now()
    # 4 Afisare rezultate separate pe categorii(False/True positiv, False/True negativ)
    print_results("Logistic Regression", test_predicted_lr, test_labels, end_train_time - start_time,
                  end_test_time - end_train_time)
    pickle.dump(lr, open("lr_model.pickle", "wb"))

    # 2 antrenare KNN
    start_time = datetime.datetime.now()
    knn = KNeighborsClassifier(n_neighbors=3)
    knn.fit(train_tfidf, train_labels)
    end_train_time = datetime.datetime.now()
    # 3 testare KNN
    test_predicted_knn = knn.predict(test_tfidf)
    end_test_time = datetime.datetime.now()
    # 4 Afisare rezultate separate pe categorii(False/True positiv, False/True negativ)
    print_results("KNN", test_predicted_knn, test_labels, end_train_time - start_time, end_test_time - end_train_time)
    pickle.dump(knn, open("knn_model.pickle", "wb"))

    # 2 antrenare AdaBoost
    start_time = datetime.datetime.now()
    ada = AdaBoostClassifier(n_estimators=100, random_state=0)
    ada.fit(train_tfidf, train_labels)
    end_train_time = datetime.datetime.now()
    # 3 testare AdaBoost
    test_predicted_ada = ada.predict(test_tfidf)
    end_test_time = datetime.datetime.now()
    # 4 Afisare rezultate separate pe categorii(False/True positiv, False/True negativ)
    print_results("AdaBoost", test_predicted_ada, test_labels, end_train_time - start_time,
                  end_test_time - end_train_time)
    pickle.dump(ada, open("ada_model.pickle", "wb"))

    # 2 antrenare RandomForest
    start_time = datetime.datetime.now()
    rforest = RandomForestClassifier(max_depth=2, random_state=0)
    rforest.fit(train_tfidf, train_labels)
    end_train_time = datetime.datetime.now()
    # 3 testare RandomForest
    test_predicted_rforest = rforest.predict(test_tfidf)
    end_test_time = datetime.datetime.now()
    # 4 Afisare rezultate separate pe categorii(False/True positiv, False/True negativ)
    print_results("RandomForest", test_predicted_rforest, test_labels, end_train_time - start_time,
                  end_test_time - end_train_time)
    pickle.dump(rforest, open("rforest_model.pickle", "wb"))

    # 2 antrenare DecisionTreeClassifier
    start_time = datetime.datetime.now()
    dtree = DecisionTreeClassifier()
    dtree.fit(train_tfidf, train_labels)
    end_train_time = datetime.datetime.now()
    # 3 testare DecisionTreeClassifier
    test_predicted_dtree = dtree.predict(test_tfidf)
    end_test_time = datetime.datetime.now()
    # 4 Afisare rezultate separate pe categorii(False/True positiv, False/True negativ)
    print_results("DecisionTree", test_predicted_dtree, test_labels, end_train_time - start_time,
                  end_test_time - end_train_time)
    pickle.dump(dtree, open("dtree_model.pickle", "wb"))

    # HardVotingAll method 1 (All above vote a category and majority win)
    # 2 antrenare HardVotingAll
    start_time = datetime.datetime.now()
    hard_voting_all = VotingClassifier(estimators=[
        ('mnb', multi_naive_bayes),
        ('pa', pa),
        ('svm', svm),
        ('lr', lr),
        ('knn', knn),
        ('ada', ada),
        ('rforest', rforest),
        ('dtree', dtree),
    ],
        voting='hard')
    hard_voting_all.fit(train_tfidf, train_labels)
    end_train_time = datetime.datetime.now()
    # 3 testare HardVotingAll
    test_predicted_hard_voting_all = hard_voting_all.predict(test_tfidf)
    end_test_time = datetime.datetime.now()
    # 4 Afisare rezultate separate pe categorii(False/True positiv, False/True negativ)
    print_results("HardVotingAll", test_predicted_hard_voting_all, test_labels, end_train_time - start_time,
                  end_test_time - end_train_time)
    pickle.dump(hard_voting_all, open("hard_voting_all.pickle", "wb"))

    # HardVotingBest3 method 1 (Best 3 vote a category and majority win)
    # 2 antrenare HardVotingAll
    start_time = datetime.datetime.now()
    hard_voting_best3 = VotingClassifier(estimators=[
        ('mnb', multi_naive_bayes),
        ('pa', pa),
        ('svm', svm)
    ],
        voting='hard')
    hard_voting_best3.fit(train_tfidf, train_labels)
    end_train_time = datetime.datetime.now()
    # 3 testare HardVotingAll
    test_predicted_hard_voting_best3 = hard_voting_best3.predict(test_tfidf)
    end_test_time = datetime.datetime.now()
    # 4 Afisare rezultate separate pe categorii(False/True positiv, False/True negativ)
    print_results("HardVotingBest3", test_predicted_hard_voting_best3, test_labels, end_train_time - start_time,
                  end_test_time - end_train_time)
    pickle.dump(hard_voting_best3, open("hard_voting_best3.pickle", "wb"))

    # SoftVotingAll method (median proability)
    # 2 antrenare SoftVotingAll
    svm.probability = True
    start_time = datetime.datetime.now()
    soft_voting_all = VotingClassifier(estimators=[
        ('mnb', multi_naive_bayes),
        ('lr', lr),
        ('knn', knn),
        ('ada', ada),
        ('rforest', rforest),
        ('dtree', dtree),
        ('svm', svm),
    ],
        voting='soft')
    soft_voting_all.fit(train_tfidf, train_labels)
    end_train_time = datetime.datetime.now()
    # 3 testare SoftVotingAll
    test_predicted_soft_voting_all = soft_voting_all.predict(test_tfidf)
    end_test_time = datetime.datetime.now()
    # 4 Afisare rezultate separate pe categorii(False/True positiv, False/True negativ)
    print_results("SoftVotingAll", test_predicted_soft_voting_all, test_labels, end_train_time - start_time,
                  end_test_time - end_train_time)
    pickle.dump(soft_voting_all, open("soft_voting_all.pickle", "wb"))

    # SoftVotingBest3 method (median proability)
    # 2 antrenare SoftVotingBest3
    start_time = datetime.datetime.now()
    svm.probability = True
    soft_voting_best3 = VotingClassifier(estimators=[
        ('mnb', multi_naive_bayes),
        ('lr', lr),
        ('svm', svm)
    ],
        voting='soft')
    soft_voting_best3.fit(train_tfidf, train_labels)
    end_train_time = datetime.datetime.now()
    # 3 testare SoftVotingBest3
    test_predicted_soft_voting_best3 = soft_voting_best3.predict(test_tfidf)
    end_test_time = datetime.datetime.now()
    # 4 Afisare rezultate separate pe categorii(False/True positiv, False/True negativ)
    print_results("SoftVotingBest3", test_predicted_soft_voting_best3, test_labels, end_train_time - start_time,
                  end_test_time - end_train_time)
    pickle.dump(soft_voting_best3, open("soft_voting_best3.pickle", "wb"))

# Algoritmi adaugati:
# 1)Multinomial Naive Bayes,
# 2)SVM ,
# 3)Passive Aggressive,
# 4)Logistic Regression,
# 5)KNN,
# 6)AdaBoost,
# 7)RandomForest,
# 8)DecisionTree
# 9)HardVotingAll(All models above vote a category and majority win by )
# 10)SoftVotingAll(All models above vote a category and majority win)
# 11)HardVotingBest3(Only top 3 models vote a category and majority win)

# Rezultate:
# variante ok :
# SVM fara stopwords si fara punctuatie dar are cel mai mare timp la antrenare
# Passive Aggressive fara stopwords si fara punctuatie are scoruri cu doar un pic mai mici dar are timp mult mai ok de antrenare

# varianta 1: tfidf fara stopword si fara punctuatie:

# Multinomial Naive Bayes Train duration: 0.01011  sec  Test duration: 0.000968  sec
# Multinomial Naive Bayes Offensive - TN: 2035 FN: 291 Precision: 87.489 % Recall: 93.606 % F1 score: 90.444190441481
# Multinomial Naive Bayes NonOffensive - TP: 2258 FP: 139 Precision: 94.201 % Recall: 88.584 % F1 score: 91.30619453456247
# Passive Aggressive Train duration: 0.075129  sec  Test duration: 0.000607  sec
# Passive Aggressive Offensive - TN: 2200 FN: 126 Precision: 94.583 % Recall: 91.324 % F1 score: 92.92493442420135
# Passive Aggressive NonOffensive - TP: 2188 FP: 209 Precision: 91.281 % Recall: 94.555 % F1 score: 92.88915985062098
# SVM Train duration: 90.786705  sec  Test duration: 5.21549  sec
# SVM Offensive - TN: 2214 FN: 112 Precision: 95.185 % Recall: 92.597 % F1 score: 93.87316617141153
# SVM NonOffensive - TP: 2220 FP: 177 Precision: 92.616 % Recall: 95.197 % F1 score: 93.8887654422218
# Logistic Regression Train duration: 0.090595  sec  Test duration: 0.000648  sec
# Logistic Regression Offensive - TN: 2170 FN: 156 Precision: 93.293 % Recall: 88.247 % F1 score: 90.69987188498403
# Logistic Regression NonOffensive - TP: 2108 FP: 289 Precision: 87.943 % Recall: 93.11 % F1 score: 90.45277051471116
# KNN Train duration: 0.00626  sec  Test duration: 1.926278  sec
# KNN Offensive - TN: 1678 FN: 648 Precision: 72.141 % Recall: 93.743 % F1 score: 81.53545565575943
# KNN NonOffensive - TP: 2285 FP: 112 Precision: 95.327 % Recall: 77.907 % F1 score: 85.74114306660356
# AdaBoost Train duration: 3.781792  sec  Test duration: 0.114046  sec
# AdaBoost Offensive - TN: 2153 FN: 173 Precision: 92.562 % Recall: 75.783 % F1 score: 83.33631585137664
# AdaBoost NonOffensive - TP: 1709 FP: 688 Precision: 71.297 % Recall: 90.808 % F1 score: 79.8783254803985
# RandomForest Train duration: 0.412585  sec  Test duration: 0.049654  sec
# RandomForest Offensive - TN: 654 FN: 1672 Precision: 28.117 % Recall: 95.614 % F1 score: 43.45521878914742
# RandomForest NonOffensive - TP: 2367 FP: 30 Precision: 98.748 % Recall: 58.604 % F1 score: 73.55518572372769
# DecisionTree Train duration: 8.43165  sec  Test duration: 0.009161  sec
# DecisionTree Offensive - TN: 2170 FN: 156 Precision: 93.293 % Recall: 87.535 % F1 score: 90.32232569071161
# DecisionTree NonOffensive - TP: 2088 FP: 309 Precision: 87.109 % Recall: 93.048 % F1 score: 89.98060838046815
# HardVotingAll Train duration: 106.199926  sec  Test duration: 6.835487  sec
# HardVotingAll Offensive - TN: 2210 FN: 116 Precision: 95.013 % Recall: 92.199 % F1 score: 93.58485125953464
# HardVotingAll NonOffensive - TP: 2210 FP: 187 Precision: 92.199 % Recall: 95.013 % F1 score: 93.58485125953464
# HardVotingBest3 Train duration: 92.643541  sec  Test duration: 5.31051  sec
# HardVotingBest3 Offensive - TN: 2212 FN: 114 Precision: 95.099 % Recall: 92.824 % F1 score: 93.94772939980736
# HardVotingBest3 NonOffensive - TP: 2226 FP: 171 Precision: 92.866 % Recall: 95.128 % F1 score: 93.98339146994054
# SoftVotingAll Train duration: 435.361925  sec  Test duration: 6.598984  sec
# SoftVotingAll Offensive - TN: 2188 FN: 138 Precision: 94.067 % Recall: 92.516 % F1 score: 93.28505353649582
# SoftVotingAll NonOffensive - TP: 2220 FP: 177 Precision: 92.616 % Recall: 94.148 % F1 score: 93.37571660491314
# SoftVotingBest3 Train duration: 410.890203  sec  Test duration: 5.295266  sec
# SoftVotingBest3 Offensive - TN: 2189 FN: 137 Precision: 94.11 % Recall: 93.507 % F1 score: 93.80753098066805
# SoftVotingBest3 NonOffensive - TP: 2245 FP: 152 Precision: 93.659 % Recall: 94.249 % F1 score: 93.95307374885581

# varianta 2: tfidf fara stopword si fara punctuatie . cu lematizare:

# Multinomial Naive Bayes Train duration: 0.009968  sec  Test duration: 0.001104  sec
# Multinomial Naive Bayes Offensive - TN: 2033 FN: 289 Precision: 87.554 % Recall: 91.825 % F1 score: 89.63865391155039
# Multinomial Naive Bayes NonOffensive - TP: 2220 FP: 181 Precision: 92.461 % Recall: 88.481 % F1 score: 90.42722796255153
# SVM Train duration: 83.177405  sec  Test duration: 5.069692  sec
# SVM Offensive - TN: 2193 FN: 129 Precision: 94.444 % Recall: 91.566 % F1 score: 92.9827353798183
# SVM NonOffensive - TP: 2199 FP: 202 Precision: 91.587 % Recall: 94.459 % F1 score: 93.00083240703914
# Passive Aggressive Train duration: 0.078349  sec  Test duration: 0.000782  sec
# Passive Aggressive Offensive - TN: 2179 FN: 143 Precision: 93.842 % Recall: 90.19 % F1 score: 91.97976417144845
# Passive Aggressive NonOffensive - TP: 2164 FP: 237 Precision: 90.129 % Recall: 93.801 % F1 score: 91.92834588158539
# Logistic Regression Train duration: 0.125628  sec  Test duration: 0.000788  sec
# Logistic Regression Offensive - TN: 2177 FN: 145 Precision: 93.755 % Recall: 87.22 % F1 score: 90.3695107058986
# Logistic Regression NonOffensive - TP: 2082 FP: 319 Precision: 86.714 % Recall: 93.489 % F1 score: 89.97414189552893
# KNN Train duration: 0.015101  sec  Test duration: 1.700548  sec
# KNN Offensive - TN: 2250 FN: 72 Precision: 96.899 % Recall: 56.59 % F1 score: 71.45156213148826
# KNN NonOffensive - TP: 675 FP: 1726 Precision: 28.113 % Recall: 90.361 % F1 score: 42.88398792984114
# AdaBoost Train duration: 3.581233  sec  Test duration: 0.1066  sec
# AdaBoost Offensive - TN: 2185 FN: 137 Precision: 94.1 % Recall: 74.093 % F1 score: 82.90655734780877
# AdaBoost NonOffensive - TP: 1637 FP: 764 Precision: 68.18 % Recall: 92.277 % F1 score: 78.41908872782118
# RandomForest Train duration: 0.404223  sec  Test duration: 0.056341  sec
# RandomForest Offensive - TN: 1995 FN: 327 Precision: 85.917 % Recall: 74.552 % F1 score: 79.83204461920995
# RandomForest NonOffensive - TP: 1720 FP: 681 Precision: 71.637 % Recall: 84.025 % F1 score: 77.33806484562706
# DecisionTree Train duration: 7.124044  sec  Test duration: 0.009006  sec
# DecisionTree Offensive - TN: 2173 FN: 149 Precision: 93.583 % Recall: 87.339 % F1 score: 90.35325319198328
# DecisionTree NonOffensive - TP: 2086 FP: 315 Precision: 86.88 % Recall: 93.333 % F1 score: 89.99096668941752

# varianta 3: tfidf fara stopword si fara punctuatie . cu lematizare si postag:

# Multinomial Naive Bayes Train duration: 0.010189  sec  Test duration: 0.001149  sec
# Multinomial Naive Bayes Offensive - TN: 2004 FN: 318 Precision: 86.305 % Recall: 92.18 % F1 score: 89.14580945177467
# Multinomial Naive Bayes NonOffensive - TP: 2231 FP: 170 Precision: 92.92 % Recall: 87.525 % F1 score: 90.1418493169664
# SVM Train duration: 98.727605  sec  Test duration: 6.130184  sec
# SVM Offensive - TN: 2168 FN: 154 Precision: 93.368 % Recall: 91.477 % F1 score: 92.41282735264681
# SVM NonOffensive - TP: 2199 FP: 202 Precision: 91.587 % Recall: 93.455 % F1 score: 92.51157126490203
# Passive Aggressive Train duration: 0.073983  sec  Test duration: 0.000786  sec
# Passive Aggressive Offensive - TN: 2152 FN: 170 Precision: 92.679 % Recall: 90.649 % F1 score: 91.65276085486123
# Passive Aggressive NonOffensive - TP: 2179 FP: 222 Precision: 90.754 % Recall: 92.763 % F1 score: 91.74750352283442
# Logistic Regression Train duration: 0.10613  sec  Test duration: 0.009857  sec
# Logistic Regression Offensive - TN: 2141 FN: 181 Precision: 92.205 % Recall: 87.316 % F1 score: 89.6939275070883
# Logistic Regression NonOffensive - TP: 2090 FP: 311 Precision: 87.047 % Recall: 92.03 % F1 score: 89.46917147372359
# KNN Train duration: 0.011524  sec  Test duration: 1.63703  sec
# KNN Offensive - TN: 2271 FN: 51 Precision: 97.804 % Recall: 55.648 % F1 score: 70.93549764095614
# KNN NonOffensive - TP: 591 FP: 1810 Precision: 24.615 % Recall: 92.056 % F1 score: 38.843559067806055
# AdaBoost Train duration: 4.064104  sec  Test duration: 0.116097  sec
# AdaBoost Offensive - TN: 2175 FN: 147 Precision: 93.669 % Recall: 72.573 % F1 score: 81.78246576677373
# AdaBoost NonOffensive - TP: 1579 FP: 822 Precision: 65.764 % Recall: 91.483 % F1 score: 76.52022629366537
# RandomForest Train duration: 0.502948  sec  Test duration: 0.058737  sec
# RandomForest Offensive - TN: 625 FN: 1697 Precision: 26.916 % Recall: 91.241 % F1 score: 41.569145391301404
# RandomForest NonOffensive - TP: 2341 FP: 60 Precision: 97.501 % Recall: 57.974 % F1 score: 72.71295030069143
# DecisionTree Train duration: 10.112135  sec  Test duration: 0.010802  sec
# DecisionTree Offensive - TN: 2148 FN: 174 Precision: 92.506 % Recall: 83.972 % F1 score: 88.03265939097223
# DecisionTree NonOffensive - TP: 1991 FP: 410 Precision: 82.924 % Recall: 91.963 % F1 score: 87.20991053651787

## varianta 4: tfidf fara stopword si fara punctuatie dar cu prepocesarea a doar catorva greseli gramaticale:

# Multinomial Naive Bayes Train duration: 0.010443  sec  Test duration: 0.00088  sec
# Multinomial Naive Bayes Offensive - TN: 1961 FN: 365 Precision: 84.308 % Recall: 92.631 % F1 score: 88.27374799224592
# Multinomial Naive Bayes NonOffensive - TP: 2241 FP: 156 Precision: 93.492 % Recall: 85.994 % F1 score: 89.58638610253725
# Passive Aggressive Train duration: 0.055143  sec  Test duration: 0.000558  sec
# Passive Aggressive Offensive - TN: 2200 FN: 126 Precision: 94.583 % Recall: 89.431 % F1 score: 91.93487748758245
# Passive Aggressive NonOffensive - TP: 2137 FP: 260 Precision: 89.153 % Recall: 94.432 % F1 score: 91.71660098591933
# SVM Train duration: 76.065727  sec  Test duration: 4.972124  sec
# SVM Offensive - TN: 2208 FN: 118 Precision: 94.927 % Recall: 90.603 % F1 score: 92.71461198727968
# SVM NonOffensive - TP: 2168 FP: 229 Precision: 90.446 % Recall: 94.838 % F1 score: 92.58994568338335
# Logistic Regression Train duration: 0.078564  sec  Test duration: 0.000571  sec
# Logistic Regression Offensive - TN: 2175 FN: 151 Precision: 93.508 % Recall: 85.765 % F1 score: 89.46928561467705
# Logistic Regression NonOffensive - TP: 2036 FP: 361 Precision: 84.94 % Recall: 93.096 % F1 score: 88.83118290682783
# KNN Train duration: 0.011767  sec  Test duration: 1.734514  sec
# KNN Offensive - TN: 2280 FN: 46 Precision: 98.022 % Recall: 56.716 % F1 score: 71.85585637658495
# KNN NonOffensive - TP: 657 FP: 1740 Precision: 27.409 % Recall: 93.457 % F1 score: 42.386823639402316
# AdaBoost Train duration: 3.173194  sec  Test duration: 0.100538  sec
# AdaBoost Offensive - TN: 2190 FN: 136 Precision: 94.153 % Recall: 72.397 % F1 score: 81.85403471630141
# AdaBoost NonOffensive - TP: 1562 FP: 835 Precision: 65.165 % Recall: 91.991 % F1 score: 76.288446066329
# RandomForest Train duration: 0.359249  sec  Test duration: 0.048229  sec
# RandomForest Offensive - TN: 1953 FN: 373 Precision: 83.964 % Recall: 71.696 % F1 score: 77.34656230245407
# RandomForest NonOffensive - TP: 1626 FP: 771 Precision: 67.835 % Recall: 81.341 % F1 score: 73.97660126293773
# DecisionTree Train duration: 8.31548  sec  Test duration: 0.010973  sec
# DecisionTree Offensive - TN: 2176 FN: 150 Precision: 93.551 % Recall: 84.472 % F1 score: 88.77998991141594
# DecisionTree NonOffensive - TP: 1997 FP: 400 Precision: 83.312 % Recall: 93.014 % F1 score: 87.89608302802763
# HardVotingAll Train duration: 89.116632  sec  Test duration: 6.466456  sec
# HardVotingAll Offensive - TN: 2262 FN: 64 Precision: 97.248 % Recall: 85.877 % F1 score: 91.20946343754267
# HardVotingAll NonOffensive - TP: 2025 FP: 372 Precision: 84.481 % Recall: 96.936 % F1 score: 90.28095730830077
# HardVotingBest3 Train duration: 77.344763  sec  Test duration: 5.091365  sec
# HardVotingBest3 Offensive - TN: 2200 FN: 126 Precision: 94.583 % Recall: 90.834 % F1 score: 92.67059894184459
# HardVotingBest3 NonOffensive - TP: 2175 FP: 222 Precision: 90.738 % Recall: 94.524 % F1 score: 92.59231479742203
# SoftVotingAll Train duration: 410.675553  sec  Test duration: 6.534043  sec
# SoftVotingAll Offensive - TN: 2233 FN: 93 Precision: 96.002 % Recall: 88.823 % F1 score: 92.27307610983361
# SoftVotingAll NonOffensive - TP: 2116 FP: 281 Precision: 88.277 % Recall: 95.79 % F1 score: 91.8801722198982
# SoftVotingBest3 Train duration: 436.318894  sec  Test duration: 5.240308  sec
# SoftVotingBest3 Offensive - TN: 2185 FN: 141 Precision: 93.938 % Recall: 91.884 % F1 score: 92.89964796418079
# SoftVotingBest3 NonOffensive - TP: 2204 FP: 193 Precision: 91.948 % Recall: 93.987 % F1 score: 92.95631996127678


