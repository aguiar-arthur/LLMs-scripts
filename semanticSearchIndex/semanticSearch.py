from datasets import load_dataset
from sentence_transformers import SentenceTransformer, util
import torch

def find_relevant_news(query):
    query_embedding = model.encode(query)
    query_embedding.shape

    similarities = util.cos_sim(query_embedding, passage_embeddings)
    top_indicies = torch.topk(similarities.flatten(), 3).indices
    top_relevant_passages = [df.iloc[x.item()]['summary'][:200] + "..." for x in top_indicies]

    return top_relevant_passages

dataset = load_dataset("multi_news", split="test")
df = dataset.to_pandas().sample(2000, random_state=42)
model = SentenceTransformer("all-MiniLM-L6-v2")
passage_embeddings = list(model.encode(df['summary'].to_list(), show_progress_bar=True))
passage_embeddings[0].shape
query = "Find me some articles about technology and artificial intelligence"

find_relevant_news("Natural disasters")
