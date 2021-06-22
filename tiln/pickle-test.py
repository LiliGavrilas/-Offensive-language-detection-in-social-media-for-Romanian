import pickle

from nltk.tokenize import word_tokenize
from string import punctuation
# from google.colab import drive
#
# drive.mount('/content/drive')
from main import procesare_comments, comments_word_conts, tfidf_transformer

# testare secundara pe cateva commenturi(pentru interfata)

# 5 Si un text cu comment-uri random pe modelul antrenat si testat anterior
multi_naive_bayes_loaded = pickle.load(open("multi_naive_bayes_model.pickle", "rb"))
pa_loaded = pickle.load(open("pa_model.pickle", "rb"))
rforest_loaded = pickle.load(open("rforest_model.pickle", "rb"))
svm_loaded = pickle.load(open("svm_model.pickle", "rb"))
dtree_loaded = pickle.load(open("dtree_model.pickle", "rb"))
knn_loaded = pickle.load(open("knn_model.pickle", "rb"))
lr_loaded = pickle.load(open("lr_model.pickle", "rb"))
ada_loaded = pickle.load(open("ada_model.pickle", "rb"))

comments_test = ["Loredana a ramas proasta", "Asta e o porcarie de emisiune", "Ce imi place emisiunea",
                 "Esti un prost"]
comments_test2 = procesare_comments(comments_test)
print(comments_test2)
test_conts = comments_word_conts.transform(comments_test2)
test_tfidf = tfidf_transformer.transform(test_conts)
test_predicted_nb = multi_naive_bayes_loaded.predict(test_tfidf)
test_predicted_pa = pa_loaded.predict(test_tfidf)
for comment, predicted_label_nb, predicted_label_pa in zip(comments_test, test_predicted_nb, test_predicted_pa):
    print("Ex input comment: \"" + comment + "\" - predicted label NB output:", predicted_label_nb, "PA output:",
          predicted_label_pa)

# testare secundara pe cateva commenturi(pentru interfata)
if 0:
    # 6 Si un text cu comment-uri random pe modelul antrenat si testat anterior din care detectam partea ce a dat hit
    multi_naive_bayes_loaded = pickle.load(open("multi_naive_bayes_model.pickle", "rb"))
    pa_loaded = pickle.load(open("pa_model.pickle", "rb"))
    rforest_loaded = pickle.load(open("rforest_model.pickle", "rb"))
    svm_loaded = pickle.load(open("svm_model.pickle", "rb"))
    dtree_loaded = pickle.load(open("dtree_model.pickle", "rb"))
    knn_loaded = pickle.load(open("knn_model.pickle", "rb"))
    lr_loaded = pickle.load(open("lr_model.pickle", "rb"))
    ada_loaded = pickle.load(open("ada_model.pickle", "rb"))

    comments_test = ["Loredana a ramas proasta", "Asta e o porcarie de emisiune", "Ce imi place emisiunea",
                     "Esti un prost", "o mare teapa nu recomand!"]

    for intial_comment in comments_test:
        comment_proccesat = procesare_comments([intial_comment])[0]
        test_cont = comments_word_conts.transform([comment_proccesat])
        test_tfidf = tfidf_transformer.transform(test_cont)
        test_predicted_nb_comment = multi_naive_bayes_loaded.predict(test_tfidf)[0]
        test_predicted_pa_comment = pa_loaded.predict(test_tfidf)[0]
        print("Intial comment:", intial_comment, "\" - predicted label NB output:", test_predicted_nb_comment,
              "PA output:", test_predicted_pa_comment)
        for word in word_tokenize(comment_proccesat):
            test_cont = comments_word_conts.transform([word])
            test_tfidf = tfidf_transformer.transform(test_cont)
            test_predicted_nb_word = multi_naive_bayes_loaded.predict(test_tfidf)[0]
            test_predicted_pa_word = pa_loaded.predict(test_tfidf)[0]

            if (test_predicted_nb_word == test_predicted_nb_comment and test_predicted_nb_comment == 1):
                print("NB A possible bad word from comment(same prediction as main comment):", word)
            if (test_predicted_nb_word == test_predicted_nb_comment and test_predicted_nb_comment == 0):
                print("NB A possible good word from comment(same prediction as main comment):", word)
            if (test_predicted_pa_word == test_predicted_pa_comment and test_predicted_pa_comment == 1):
                print("PA A possible bad word from comment(same prediction as main comment):", word)
            if (test_predicted_pa_word == test_predicted_pa_comment and test_predicted_pa_comment == 0):
                print("PA A possible good word from comment(same prediction as main comment):", word)

