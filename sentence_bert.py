import json
import os
from sentence_transformers import SentenceTransformer, util
embedder = SentenceTransformer('msmarco-distilbert-base-v4')

def read_text_from_file(file_name):
    file_name = "./data/coref_text_files/" + file_name + ".txt"
    input_file = open(file_name, 'r')
    text = ""
    for line in input_file:
        text += line
    return text


def get_text_similarity_score(pair):
    text1, text2 = read_text_from_file(pair[0]), read_text_from_file(pair[1])
    texts = [text1, text2]
    embeddings = embedder.encode(texts)
    e1, e2 = embeddings[0], embeddings[1]
    similarity_score = util.pytorch_cos_sim(e1, e2)
    return round(similarity_score.item(), 3)
