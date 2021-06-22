from transformers import AutoTokenizer, AutoModel
import torch
# load tokenizer and model
tokenizer = AutoTokenizer.from_pretrained("dumitrescustefan/bert-base-romanian-cased-v1")
model = AutoModel.from_pretrained("dumitrescustefan/bert-base-romanian-cased-v1")
# tokenize a sentence and run through the model
input_ids = torch.tensor(tokenizer.encode("Acest test.", add_special_tokens=True)).unsqueeze(0)  # Batch size 1
outputs = model(input_ids)
# get encoding
last_hidden_states = outputs[0]  # The last hidden-state is the first element of the output tuple


print(input_ids)

tokenized_text = tokenizer.tokenize("Acesta este un test.")

indexed_tokens = tokenizer.convert_tokens_to_ids(tokenized_text)

for tup in zip(tokenized_text, indexed_tokens):
    print('{:<12} {:>6,}'.format(tup[0], tup[1]))

segments_ids = [1] * len(tokenized_text)
print (last_hidden_states)


