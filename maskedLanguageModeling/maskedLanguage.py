from transformers import AutoTokenizer, AutoModelForMaskedLM
from scipy.special import softmax
import numpy as np

model_name = "bert-base-cased"

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForMaskedLM.from_pretrained(model_name)
mask = tokenizer.mask_token

sentence = f"I want to {mask} a pizza for tonight."

tokens = tokenizer.tokenize(sentence)

encoded_inputs = tokenizer(sentence, return_tensors='pt')
outputs = model(**encoded_inputs)
logits = outputs.logits.detach().numpy()[0]

mask_logits = logits[tokens.index(mask) + 1]
confidence_scores = softmax(mask_logits)

# precdiction
for i in np.argsort(confidence_scores)[::-1][:5]:
    pred_token = tokenizer.decode(i)
    score = confidence_scores[i]

    print(sentence.replace(mask, pred_token), score)
