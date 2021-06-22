import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.corpus import wordnet as wn_nltk_test

from string import punctuation


#install nltk stopwords:
#nltk.download('stopwords')
#print(stopwords.words('romanian'))

#install nltk lemmatizer:
#nltk.download('wordnet')

#nltk.download()
#print(nltk.__version__)


#stemmer RO : https://pypi.org/project/snowballstemmer/
# merge dar are niste rezultate nu foarte ok. Parca nu taie tot timpul corect extensiile
import snowballstemmer

#lematizare RO v1 : https://github.com/dumitrescustefan/RoWordNet
# merge dar are niste rezultate nu foarte ok. Nu gaseste tot timpul cuvantul. 
import rowordnet as rwn

#lematizare RO v2 : https://pypi.org/project/wn/
# merge dar are niste rezultate nu foarte ok. Nu gaseste tot timpul cuvantul
import wn

#lematizare RO v3 : from nltk.stem.wordnet import WordNetLemmatizer 
# nu am gasit versiunea de romana : ar putea fi 
from nltk.stem.wordnet import WordNetLemmatizer 

def read_corpus_file(corpus_file_path):
    """ Functie de citire csv corpus si spargere in vectori de categorii cu toate commenturile din aceiasi categorie""" 
    """ Csv de forma asta pe linie: "categorie,comment" """
    """ categorie=0 ->offensive categorie=1 ->neutru categorie=2 ->nonoffensive """
    
    corpus = open(corpus_file_path, 'r')

    linie_gresita = 0
    
    corpus_offensive = []
    corpus_nonoffensive = []
    corpus_neutru = []
    
    for linie in corpus:
        #scoatem enter
        linie = linie.strip()
        #spargem doar dupa prima virgula
        linie_sparta = linie.split(",", 1) 
        if len(linie_sparta) == 2:
            categorie , comment = linie_sparta
            if (categorie == "1"):
                corpus_offensive.append(comment)
            elif (categorie == "2"):
                corpus_neutru.append(comment)
            elif (categorie == "0"):
                corpus_nonoffensive.append(comment)
            else:
                print("Linie gresita tip1 : categorie=" , categorie , "comment=",comment)
                linie_gresita = linie_gresita + 1
        else:
            print("Linie gresita tip2:" , linie)
            linie_gresita = linie_gresita + 1
        total = len(corpus_offensive) + len(corpus_nonoffensive) + len(corpus_neutru) + linie_gresita
        
    print("Avem" , total , " linii citite corect si " , linie_gresita , " incorecte in fisier ")
    print(" Offensive =" , len(corpus_offensive) , "\n Nonoffensive =" , len(corpus_nonoffensive) , "\n Neutru =" , len(corpus_neutru))
    return corpus_offensive , corpus_nonoffensive , corpus_neutru

#offensive , nonoffensive , neutru = read_corpus_file('corpus_final.csv')
#print("Corpus offensinve" , offensive)
#print("Corpus nonoffensive" , nonoffensive)
#print("Corpus neutru" , neutru)



examplu_propozitie_ro = """tu.Adrian"""
examplu_propozitie_en = """This is a sample sentence, showing off the stop words filtration. rocks"""
if(1):
	
	limba = 1 # 1 RO 0 EN
	if limba == 1:
		propozitie_tokenizata = word_tokenize(examplu_propozitie_ro)  #tokenizarea merge in romana
		print("Fraza ro tokenizata ",propozitie_tokenizata)

		stop_words_ro = set(stopwords.words('romanian'))
		propozitie_tokenizata_fara_stopwords = [w for w in propozitie_tokenizata if not w in stop_words_ro]  #scoaterea de stopwords merge si in romana
		print("Fraza ro tokenizata_fara_stopwords ",propozitie_tokenizata_fara_stopwords)
		
		propozitie_tokenizata_fara_stopwords_si_puncte = [w for w in propozitie_tokenizata_fara_stopwords if not w in punctuation]
		print("Fraza ro tokenizata_fara_stopwords_si_puncte ",propozitie_tokenizata_fara_stopwords_si_puncte)
		
		#lem = WordNetLemmatizer()
		#lemmatized_sentence_en = [lem.lemmatize(w) for w in filtered_sentence]  #lematizare in engleza
		#print("Fraza lemmatizata engleza",lemmatized_sentence_en)#nu am reusit sa il fac sa mearga si in romana
		
		stemmer_ro = snowballstemmer.stemmer('romanian');
		propozitie_tokenizata_fara_stopwords_si_puncte_dupa_stemmer = stemmer_ro.stemWords(propozitie_tokenizata_fara_stopwords_si_puncte)
		print("Fraza dupa stemmer in ro ", propozitie_tokenizata_fara_stopwords_si_puncte_dupa_stemmer)
	
	else:
		word_tokens = word_tokenize(examplu_propozitie_en)  #tokenizarea merge si in romana
		print("Fraza en tokenizata ",word_tokens)

		stop_words_en = set(stopwords.words('english'))
		filtered_sentence = [w for w in word_tokens if not w in stop_words_en]  
		print("Fraza fara stopwords ", filtered_sentence)
		
		filtered_sentence_without_punctuation = [w for w in filtered_sentence if not w in punctuation]
		print("Fraza en tokenizata_fara_stopwords ",filtered_sentence_without_punctuation)
		

		lem = WordNetLemmatizer()
		lemmatized_sentence_en = [lem.lemmatize(w) for w in filtered_sentence_without_punctuation]  #lematizare in engleza
		print("Fraza lemmatizata engleza", lemmatized_sentence_en)#nu am reusit sa il fac sa mearga si in romana

		stemmer_en = snowballstemmer.stemmer('english');
		stemmer_sentence_en = stemmer_en.stemWords(lemmatized_sentence_en)
		print("Fraza dupa stemmer in en ",stemmer_sentence_en)
			


else:#teste:
	#posibila alternativa lematizare in romana : https://github.com/dumitrescustefan/RoWordNet
	wn = rwn.RoWordNet()
	cuvant_initial = 'carte'
	#stemmer_ro = snowballstemmer.stemmer('romanian');
	# stemmer_sentence_ro = stemmer_ro.stemWords([cuvant_initial])
	# print(stemmer_sentence_ro)
	# synset_ids = wn.synsets(literal=stemmer_sentence_ro[0])
	synset_ids = wn.synsets(literal=cuvant_initial)
	if len(synset_ids) >= 1 :
		for synset_id in synset_ids:
			print("Posibila lematizare pt ",cuvant_initial,": literals=", wn(synset_id).literals ," tip=", wn(synset_id).pos)
	else:
		print("NU are lematizare in acest modul: ",cuvant_initial)


	#wn.download('ronwn')
	#w = wn.words('arbusti')[0]
	#print(w.lemma())
	#nltk.download()
	#print("NLTK wordnet languages:",  sorted(wn_nltk_test.langs()))




