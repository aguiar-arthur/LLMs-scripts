from transformers import BertModel, AutoTokenizer
from scipy.spatial.distance import cosine
import matplotlib.pyplot as plt

def predict(text):
    encoded_inputs = tokenizer(text, return_tensors="pt")

    return model(**encoded_inputs)[0]

model_name = "bert-base-cased"
model = BertModel.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

sentence1 = "There was a fly drinking from my soup"
sentence2 = "There is a fly swimming in my juice"

tokens1 = tokenizer.tokenize(sentence1)
tokens2 = tokenizer.tokenize(sentence2)

out1 = predict(sentence1)
out2 = predict(sentence2)

emb1 = out1[0:, tokens1.index("fly"), :].detach()
emb2 = out2[0:, tokens2.index("fly"), :].detach()

emb1.shape
emb2.shape

similarity = cosine(emb1[0], emb2[0])

print("Cosine Similarity:", similarity)

plt.scatter(emb1[0], emb2[0], color='blue')
plt.title('Spatial Distance between Embeddings')
plt.xlabel('Embedding of "fly" in Sentence 1')
plt.ylabel('Embedding of "fly" in Sentence 2')
plt.show()
