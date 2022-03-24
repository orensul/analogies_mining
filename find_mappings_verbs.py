from sklearn.cluster import AgglomerativeClustering
import numpy as np
from sentence_transformers import SentenceTransformer, util
import networkx as nx
import matplotlib.pyplot as plt

from qa_srl import read_parsed_qasrl_verbs

verbs_similarity_embedder = SentenceTransformer('msmarco-distilbert-base-v4')
clustering_embedder = verbs_similarity_embedder

# clustering_embedder = SentenceTransformer('msmarco-distilbert-base-v4')


max_num_words_in_entity = 7
score_boosting_same_sentence = 1
beam = 7
max_mappings_before_beam_search = 20
verbose = False

def print_corpus_entities(text1_corpus_entities, text2_corpus_entities):
    print("text1 entities:")
    for entity in text1_corpus_entities:
        print(entity)
    print()
    print("text2 entities:")
    for entity in text2_corpus_entities:
        print(entity)
    print()

def get_clusters_of_entities(answer_verb_map, corpus_entities, distance_threshold):
    filtered_corpus_entities = []
    filtered_answer_verb_map = {}
    for entity in corpus_entities:
        if len(entity.split(' ')) <= max_num_words_in_entity:
            filtered_answer_verb_map[entity] = answer_verb_map[entity]
            filtered_corpus_entities.append(entity)

    corpus_entities, answer_verb_map = filtered_corpus_entities, filtered_answer_verb_map

    # for AgglomerativeClustering clustering at least two entities are needed
    if len(corpus_entities) == 1:
        corpus_entities.append(corpus_entities[0])

    corpus_embeddings = clustering_embedder.encode(corpus_entities)

    # Normalize the embeddings to unit length
    corpus_embeddings = corpus_embeddings / np.linalg.norm(corpus_embeddings, axis=1, keepdims=True)

    # Perform kmean clustering
    clustering_model = AgglomerativeClustering(n_clusters=None,
                                               distance_threshold=distance_threshold)  # , affinity='cosine', linkage='average', distance_threshold=0.4)

    clustering_model.fit(corpus_embeddings)
    cluster_assignment = clustering_model.labels_

    clustered_sentences = {}
    for sentence_id, cluster_id in enumerate(cluster_assignment):
        if cluster_id not in clustered_sentences:
            clustered_sentences[cluster_id] = []

        clustered_sentences[cluster_id].append(corpus_entities[sentence_id])

    clusters_result = []
    for i, cluster in clustered_sentences.items():
        clusters_result.append((i + 1, set(cluster)))

    clusters_result.sort(key=lambda x: x[0])
    return clusters_result



def get_clusters_of_verbs(clusters_of_entities, answer_verb_map):
    clusters_of_verbs = []
    for tup in clusters_of_entities:
        idx = tup[0]
        entities = tup[1]
        verbs = []
        for entity in entities:
            list_of_questions = answer_verb_map[entity]
            for v in list_of_questions:
                verbs.append(v)
        clusters_of_verbs.append((idx, set(verbs)))
    return clusters_of_verbs

def print_clusters_of_entities(text1_clusters_of_entities, text2_clusters_of_entities):
    print("text1 clusters of entities: ")
    for entity in text1_clusters_of_entities:
        print(entity)
    print()
    print("text2 clusters of entities: ")
    for entity in text2_clusters_of_entities:
        print(entity)
    print()


def print_clusters_of_verbs(text1_clusters_of_verbs, text2_clusters_of_verbs):
    print("text1 clusters of questions: ")
    for question in text1_clusters_of_verbs:
        print(question)
    print()
    print("text2 clusters of questions: ")
    for question in text2_clusters_of_verbs:
        print(question)
    print()

def get_sent_bert_similarity_map_between_verbs(verbs1, verbs2):
    sent_bert_similarity_map = {}
    cos_sim_all_questions_result = get_cosine_sim_between_verbs(verbs1, verbs2)
    for triplet in cos_sim_all_questions_result:
        cos_sim = round(triplet[0].tolist(), 3)
        sent_bert_similarity_map[(triplet[1], triplet[2])] = cos_sim
    return sent_bert_similarity_map


def get_cosine_sim_between_verbs(verbs1, verbs2):
    result = []

    embeddings1 = verbs_similarity_embedder.encode(verbs1, convert_to_tensor=True)
    embeddings2 = verbs_similarity_embedder.encode(verbs2, convert_to_tensor=True)
    cosine_scores = util.pytorch_cos_sim(embeddings1, embeddings2)
    # Output the pairs with their score
    for i in range(len(verbs1)):
        for j in range(len(verbs2)):
            result.append((cosine_scores[i][j], verbs1[i], verbs2[j]))
    result.sort(key=lambda x: x[0], reverse=True)
    return result


def get_entities_similarity_score(sentBert_similarity_map, verbs1, verbs2, cos_sim_threshold):
    total_score = 0
    similar_verbs = []
    for i in range(len(verbs1)):
        for j in range(len(verbs2)):
            if (verbs1[i], verbs2[j]) in sentBert_similarity_map:
                curr_score = sentBert_similarity_map[(verbs1[i], verbs2[j])]
                if curr_score >= cos_sim_threshold:
                    similar_verbs.append((verbs1[i], verbs2[j]))
                    total_score += curr_score
    return round(total_score, 3), similar_verbs


def get_verbs_list_from_cluster(cluster_of_verbs):
    return [q for tup in cluster_of_verbs for q in tup[1]]

def get_sorted_entities_clusters_scores(text1_clusters_of_entities, text1_clusters_of_verbs, text2_clusters_of_entities,
                                 text2_clusters_of_verbs, cos_sim_threshold):
    verbs1 = get_verbs_list_from_cluster(text1_clusters_of_verbs)
    verbs2 = get_verbs_list_from_cluster(text2_clusters_of_verbs)
    sent_bert_similarity_map = get_sent_bert_similarity_map_between_verbs(verbs1, verbs2)

    clusters_scores = []
    total_count, pass_threshold_count = 0, 0

    for i in range(len(text1_clusters_of_verbs)):
        for j in range(len(text2_clusters_of_verbs)):
            verbs1 = list(text1_clusters_of_verbs[i][1])
            verbs2 = list(text2_clusters_of_verbs[j][1])
            curr_score, similar_verbs = get_entities_similarity_score(sent_bert_similarity_map, verbs1,
                                                                      verbs2, cos_sim_threshold)
            total_count += 1
            if curr_score > 0:
                pass_threshold_count += 1
                clusters_scores.append(((i + 1, text1_clusters_of_entities[i][1]),
                                        (j + 1, text2_clusters_of_entities[j][1]), similar_verbs, curr_score))

    clusters_scores = sorted(clusters_scores, key=lambda t: t[::-1], reverse=True)
    return clusters_scores

def get_mappings_without_duplicates(mappings):
    seen_tuples = {}
    mappings_no_duplicates = []
    for mapping in mappings:
        if (mapping[0][0], mapping[1][0]) in seen_tuples:
            continue
        mappings_no_duplicates.append(mapping)
        seen_tuples[(mapping[0][0], mapping[1][0])] = True
    return mappings_no_duplicates


def get_consistent_mapping_indices(mapping_indices, mappings_no_dups):
    total_score = 0
    seen_base, seen_target = {}, {}
    res_of_mapping_indices = set()
    for mapping_idx in mapping_indices:
        mapping = mappings_no_dups[mapping_idx]
        if mapping[0][0] in seen_base or mapping[1][0] in seen_target:
            continue
        seen_base[mapping[0][0]] = True
        seen_target[mapping[1][0]] = True
        total_score += mapping[-1]
        res_of_mapping_indices.add(mapping_idx)
    return res_of_mapping_indices, total_score



def beam_search(B, mappings_no_dups):
    cache = [({i}, mappings_no_dups[i][-1]) for i in range(B)]
    while True:
        start_cache = cache.copy()
        for tup in cache:
            mapping_indices = tup[0]
            for j in range(len(mappings_no_dups)):
                if j in mapping_indices:
                    continue
                new_mapping_indices, curr_score = get_consistent_mapping_indices(mapping_indices.union({j}), mappings_no_dups)
                if (new_mapping_indices, curr_score) not in cache:
                    cache.append((new_mapping_indices, curr_score))
        cache = sorted(cache, key=lambda t: t[1], reverse=True)
        cache = cache[:B]
        if cache == start_cache:
            break
    return cache

def generate_mappings(pair, cos_sim_threshold):
    text1_answer_verb_map = read_parsed_qasrl_verbs(pair[0])
    text2_answer_verb_map = read_parsed_qasrl_verbs(pair[1])

    text1_corpus_entities = list(text1_answer_verb_map.keys())
    text2_corpus_entities = list(text2_answer_verb_map.keys())

    if not text1_corpus_entities or not text2_corpus_entities:
        return None, None, None
    if verbose:
        print_corpus_entities(text1_corpus_entities, text2_corpus_entities)

    text1_clusters_of_entities = get_clusters_of_entities(text1_answer_verb_map, text1_corpus_entities, distance_threshold=1)

    text1_clusters_of_verbs = get_clusters_of_verbs(text1_clusters_of_entities, text1_answer_verb_map)

    text2_clusters_of_entities = get_clusters_of_entities(text2_answer_verb_map, text2_corpus_entities,
                                                          distance_threshold=1)

    text2_clusters_of_verbs = get_clusters_of_verbs(text2_clusters_of_entities, text2_answer_verb_map)

    if verbose:
        print_clusters_of_entities(text1_clusters_of_entities, text2_clusters_of_entities)
        print_clusters_of_verbs(text1_clusters_of_verbs, text2_clusters_of_verbs)

    clusters_scores = get_sorted_entities_clusters_scores(text1_clusters_of_entities, text1_clusters_of_verbs,
                                                   text2_clusters_of_entities, text2_clusters_of_verbs, cos_sim_threshold)

    mappings_no_duplicates = get_mappings_without_duplicates(clusters_scores)
    mappings_no_duplicates = mappings_no_duplicates[:max_mappings_before_beam_search]
    if verbose:
        print("all mappings without duplicates:")
        for mapping in mappings_no_duplicates:
            print(mapping)
    if len(mappings_no_duplicates) == 0:
        return None, None, None

    top1_solution, top2_solution, top3_solution = None, None, None
    cache = beam_search(min(len(mappings_no_duplicates), beam), mappings_no_duplicates)
    if len(cache) > 0:
        top1_solution = [mappings_no_duplicates[mapping_id] for mapping_id in cache[0][0]]
        top1_solution = sorted(top1_solution, key=lambda t: t[::-1], reverse=True)
        print()
        print("top 1 solution: ")
        print(top1_solution)
    if len(cache) > 1:
        top2_solution = [mappings_no_duplicates[mapping_id] for mapping_id in cache[1][0]]
        top2_solution = sorted(top2_solution, key=lambda t: t[::-1], reverse=True)
        print("top 2 solution: ")
        print(top2_solution)

    if len(cache) > 2:
        top3_solution = [mappings_no_duplicates[mapping_id] for mapping_id in cache[2][0]]
        top3_solution = sorted(top3_solution, key=lambda t: t[::-1], reverse=True)
        print("top 3 solution: ")
        print(top3_solution)

    return top1_solution, top2_solution, top3_solution
