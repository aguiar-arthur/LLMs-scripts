from transformers import BertModel, AutoTokenizer
import pandas as pd

model_name = "bert-base-cased"
model = BertModel.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

test_sentence = "Artificial intelligence is revolutionizing the way we live."

tokens = tokenizer.tokenize(test_sentence)

vocab = tokenizer.vocab
vocab_df = pd.DataFrame({"token": vocab.keys(), "token_id": vocab.values()})
vocab_df = vocab_df.sort_values(by="token_id").set_index("token_id")

token_ids = tokenizer.encode(test_sentence)

print("Tokens:", tokens)
print("Token IDs:", token_ids)

print("Token at index 101:", vocab_df.iloc[101])
print("Token at index 102:", vocab_df.iloc[102])

print("Zipped Tokens and Token IDs:", list(zip(tokens, token_ids[1:-1])))

decoded_sentence = tokenizer.decode(token_ids[1:-1])
print("Decoded Sentence:", decoded_sentence)

