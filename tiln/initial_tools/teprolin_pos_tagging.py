import json

import requests


url = 'http://relate.racai.ro:5000/process'
d = {'text': 'pt ke?',
     'exec': 'lemmatization'}

fil = requests.post(url, data = d)
x = json.loads(fil.text)

lemmatized_sentence_ro = [v for i in x['teprolin-result']['tokenized'][0] for (k, v) in i.items() if k == "_lemma"]

print(lemmatized_sentence_ro)
