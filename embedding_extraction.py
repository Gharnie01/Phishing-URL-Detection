
import gensim.downloader as api
import pandas as pd
import numpy as np
import re

# Load FastText vector model from Gensim API
ft_model = api.load("fasttext-wiki-news-subwords-300")
print("✅ Loaded FastText model (300d)")

def url_to_words(url):
    url = url.lower()
    return [t for t in re.split(r'[^a-z0-9]+', url) if t]

def get_embedding(url, model):
    tokens = url_to_words(url)
    vectors = [model[t] for t in tokens if t in model]
    if len(vectors) == 0:
        return np.zeros(model.vector_size)
    return np.mean(vectors, axis=0)

df = pd.read_csv("data/final_selected_features_data.csv")
urls = df['url'].tolist()

print("🔄 Generating FastText embeddings...")
embeddings = np.array([get_embedding(url, ft_model) for url in urls], dtype=np.float32)
np.save("Embeddings/fasttext_embeddings.npy", embeddings)
print("✅ Embeddings saved: output/fasttext_embeddings.npy")
print("Shape:", embeddings.shape)  # (num_urls, 300)

#----------------------GLOVE----------------------

# Load GloVe vectors (300-dimensional)
glove_model = api.load("glove-wiki-gigaword-300")
print("✅ GloVe model loaded.")

def url_to_words(url):
    url = url.lower()
    return [t for t in re.split(r'[^a-z0-9]+', url) if t]

def get_embedding(url, model):
    tokens = url_to_words(url)
    vectors = [model[t] for t in tokens if t in model]
    if len(vectors) == 0:
        return np.zeros(model.vector_size)
    return np.mean(vectors, axis=0)
# Load URLs
df = pd.read_csv("data/final_selected_features_data.csv")
urls = df['url'].tolist()

print("🔄 Generating GloVe embeddings...")
glove_embeddings = np.array([get_embedding(url, glove_model) for url in urls], dtype=np.float32)

# Save
np.save("Embeddings/glove_embeddings.npy", glove_embeddings)
print("✅ Saved GloVe embeddings: Embeddings/glove_embeddings.npy")
print("Shape:", glove_embeddings.shape)

import tensorflow as tf
import tensorflow_hub as hub
import numpy as np
import pandas as pd
import os
from tqdm import tqdm

# Load dataset
df = pd.read_csv('data/final_selected_features_data.csv')
urls = df['url'].astype(str).tolist()

# Load ELMo from TF Hub
elmo_model = hub.load("https://tfhub.dev/google/elmo/3")

# Get embeddings function
def elmo_embed_batch(batch):
    return elmo_model.signatures["default"](tf.constant(batch))["elmo"].numpy()[:, 0, :]  # [CLS]-like

# Batch embedding extraction
def get_elmo_embeddings(urls, batch_size=64):
    embeddings = []
    for i in tqdm(range(0, len(urls), batch_size)):
        batch = urls[i:i + batch_size]
        embeddings.append(elmo_embed_batch(batch))
    return np.vstack(embeddings)

# Save to disk
elmo_embeddings = get_elmo_embeddings(urls)
np.save("Embeddings/elmo_embeddings.npy", elmo_embeddings)
print("✅ Saved ELMo embeddings: ", elmo_embeddings.shape)
print("Shape:", glove_embeddings.shape)

#----------------------BERT----------------------#
from transformers import DistilBertTokenizer, TFDistilBertModel
import numpy as np

df = pd.read_csv('data/final_selected_features_data.csv')
urls = df['url'].astype(str).tolist()

tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")
bert_model = TFDistilBertModel.from_pretrained("distilbert-base-uncased")
#TUNING 0: Adjusted max_len from 64 to 200 and batch_size from 256 to 128
def get_bert_embeddings(urls, tokenizer, model, max_len=200, batch_size=128):
    all_embeddings = []
    for i in range(0, len(urls), batch_size):
        batch_urls = urls[i:i+batch_size]
        encodings = tokenizer(batch_urls, truncation=True, padding='max_length',
                              max_length=max_len, return_tensors='tf')
        outputs = model(encodings['input_ids'], attention_mask=encodings['attention_mask'])
        cls_embeddings = outputs.last_hidden_state[:, 0, :]  # CLS token
        all_embeddings.append(cls_embeddings.numpy())
    return np.vstack(all_embeddings)

bert_embeddings = get_bert_embeddings(urls, tokenizer, bert_model)
np.save("Embeddings/bert_embeddings.npy", bert_embeddings)
print("✅ Saved BERT embeddings: Embeddings/bert_embeddings.npy")
print("Shape:", bert_embeddings.shape)  # (num_urls, 768)
