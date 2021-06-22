import nltk

nltk.download('punkt')
nltk.download('stopwords')
import re
import pickle
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer

from nltk.corpus import stopwords
from stop_words import get_stop_words
from nltk.tokenize import word_tokenize
from string import punctuation


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


stop_words_ro = get_stop_words('ro')

# train_comments, test_comments, train_labels, test_labels = train_test_split(all_comments_processed, all_lables,
#                                                                                 test_size=0.2, random_state=0)
#
# comments_word_conts = CountVectorizer(stop_words=stop_words_ro)
# train_data = comments_word_conts.fit_transform(train_comments)
# tfidf_transformer = TfidfTransformer(sublinear_tf=True)
# train_tfidf = tfidf_transformer.fit_transform(train_data)
#
# test_data = comments_word_conts.transform(test_comments)
# test_tfidf = tfidf_transformer.transform(test_data)

def test_comment(comment_test, model):
    # comment_test = "esti cel mai bun" #@param {type:"string"}
    # model = "soft_voting_best3"
    print("Model selectat = ", model, " and test coment \"", comment_test,"\"")

    multi_naive_bayes_loaded = pickle.load(open("multi_naive_bayes_model.pickle", "rb"))
    pa_loaded = pickle.load(open("pa_model.pickle", "rb"))
    rforest_loaded = pickle.load(open("rforest_model.pickle", "rb"))
    svm_loaded = pickle.load(open("svm_model.pickle", "rb"))
    dtree_loaded = pickle.load(open("dtree_model.pickle", "rb"))
    knn_loaded = pickle.load(open("knn_model.pickle", "rb"))
    lr_loaded = pickle.load(open("lr_model.pickle", "rb"))
    ada_loaded = pickle.load(open("ada_model.pickle", "rb"))
    hard_voting_all_loaded = pickle.load(open("hard_voting_all.pickle", "rb"))
    hard_voting_best3_loaded = pickle.load(open("hard_voting_best3.pickle", "rb"))
    soft_voting_all_loaded = pickle.load(open("soft_voting_all.pickle", "rb"))
    soft_voting_best3_loaded = pickle.load(open("soft_voting_best3.pickle", "rb"))

    stop_words_ro = get_stop_words('ro')

    # comments_word_conts = CountVectorizer(stop_words=stop_words_ro)
    # train_data = comments_word_conts.fit_transform(train_comments)
    # tfidf_transformer = TfidfTransformer(sublinear_tf=True)
    # train_tfidf = tfidf_transformer.fit_transform(train_data)
    #
    # test_data = comments_word_conts.transform(test_comments)
    # test_tfidf = tfidf_transformer.transform(test_data)


    tfidf_transformer = TfidfTransformer(sublinear_tf=True)
    comments_word_conts = CountVectorizer(stop_words=stop_words_ro, decode_error="replace", vocabulary=pickle.load(open("feature.pkl", "rb")))

    comment_proccesat = procesare_comments([comment_test])[0]
    test_tfidf = tfidf_transformer.fit_transform(comments_word_conts.fit_transform([comment_proccesat]))




    predicted_output = ""
    if model == "multi_naive_bayes_model":
      predicted_output = multi_naive_bayes_loaded.predict(test_tfidf)[0]
    if model == "rforest_model":
      predicted_output = rforest_loaded.predict(test_tfidf)[0]
    if model == "pa_model":
      predicted_output = pa_loaded.predict(test_tfidf)[0]
    if model == "svm_model":
      predicted_output = svm_loaded.predict(test_tfidf)[0]
    if model == "dtree_model":
      predicted_output = dtree_loaded.predict(test_tfidf)[0]
    if model == "knn_model":
      predicted_output = knn_loaded.predict(test_tfidf)[0]
    if model == "lr_model":
      predicted_output = lr_loaded.predict(test_tfidf)[0]
    if model == "ada_model":
      predicted_output = ada_loaded.predict(test_tfidf)[0]
    if model == "hard_voting_all":
      predicted_output = hard_voting_all_loaded.predict(test_tfidf)[0]
    if model == "hard_voting_best3":
      predicted_output = hard_voting_best3_loaded.predict(test_tfidf)[0]
    if model == "soft_voting_all":
      predicted_output = soft_voting_all_loaded.predict(test_tfidf)[0]
    if model == "soft_voting_best3":
      predicted_output = soft_voting_best3_loaded.predict(test_tfidf)[0]
    if (predicted_output == 1):
      output = "Output: Offensive"
    else:
      output = "Output: NonOffensive"

    impact_words = []
    for word in word_tokenize(comment_proccesat):
      test_cont = comments_word_conts.transform([word])
      test_tfidf = tfidf_transformer.transform(test_cont)
      predicted_word_output = ""
      if model == "multi_naive_bayes_model":
          predicted_word_output = multi_naive_bayes_loaded.predict(test_tfidf)[0]
      if model == "rforest_model":
          predicted_word_output = rforest_loaded.predict(test_tfidf)[0]
      if model == "pa_model":
          predicted_word_output = pa_loaded.predict(test_tfidf)[0]
      if model == "svm_model":
          predicted_word_output = svm_loaded.predict(test_tfidf)[0]
      if model == "dtree_model":
          predicted_word_output = dtree_loaded.predict(test_tfidf)[0]
      if model == "knn_model":
          predicted_word_output = knn_loaded.predict(test_tfidf)[0]
      if model == "lr_model":
          predicted_word_output = lr_loaded.predict(test_tfidf)[0]
      if model == "ada_model":
          predicted_word_output = ada_loaded.predict(test_tfidf)[0]
      if model == "hard_voting_all":
        predicted_word_output = hard_voting_all_loaded.predict(test_tfidf)[0]
      if model == "hard_voting_best3":
        predicted_word_output = hard_voting_best3_loaded.predict(test_tfidf)[0]
      if model == "soft_voting_all":
        predicted_word_output = soft_voting_all_loaded.predict(test_tfidf)[0]
      if model == "soft_voting_best3":
        predicted_word_output = soft_voting_best3_loaded.predict(test_tfidf)[0]
      if(predicted_word_output == predicted_output and predicted_output == 1):
        impact_words.append(word) #bad
      if(predicted_word_output == predicted_output and predicted_output == 0):
          impact_words.append(word) #good

    return (output, impact_words)