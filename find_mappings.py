
from sklearn.cluster import AgglomerativeClustering
import numpy as np
from sentence_transformers import SentenceTransformer, util
import networkx as nx
import matplotlib.pyplot as plt

from qa_srl import read_parsed_qasrl

questions_similarity_embedder = SentenceTransformer('msmarco-distilbert-base-v4')
clustering_embedder = questions_similarity_embedder

# clustering_embedder = SentenceTransformer('msmarco-distilbert-base-v4')


max_num_words_in_entity = 7
cos_sim_threshold = 0.65
entities_mapping_final_threshold = cos_sim_threshold
num_mappings_to_show = 5
beam = 7
verbose = True



def generate_mappings(pair):
    text1_answer_question_map = read_parsed_qasrl(pair[0])
    text2_answer_question_map = read_parsed_qasrl(pair[1])
    text1_corpus_entities = [key for key in text1_answer_question_map.keys()]
    text2_corpus_entities = [key for key in text2_answer_question_map.keys()]

    if verbose:
        print_corpus_entities(text1_corpus_entities, text2_corpus_entities)

    text1_clusters_of_entities = get_clustering_result(text1_answer_question_map, text1_corpus_entities, distance_threshold=1)

    text1_clusters_of_questions = get_clusters_of_questions(text1_clusters_of_entities, text1_answer_question_map)

    text2_clusters_of_entities = get_clustering_result(text2_answer_question_map, text2_corpus_entities, distance_threshold=1)

    text2_clusters_of_questions = get_clusters_of_questions(text2_clusters_of_entities, text2_answer_question_map)

    if verbose:
        print_clusters_of_entities(text1_clusters_of_entities, text2_clusters_of_entities)
        print_clusters_of_questions(text1_clusters_of_questions, text2_clusters_of_questions)

    clusters_scores = get_entities_clusters_scores(text1_clusters_of_entities, text1_clusters_of_questions,
                                                   text2_clusters_of_entities, text2_clusters_of_questions)
    clusters_scores = sorted(clusters_scores, key=lambda t: t[::-1], reverse=True)
    if verbose:
        print()
        print("sorted clusters scores:")
        print(clusters_scores)
        print()
        print("entities mappings before coreference:")
        print()

    extended_mappings = get_extended_mappings_from_clusters_scores(clusters_scores, text1_clusters_of_questions,
                                               text2_clusters_of_questions, text1_clusters_of_entities,
                                               text2_clusters_of_entities)

    if verbose:
        print("all mappings:")
        for mapping in extended_mappings:
            print(mapping)

    mappings_no_duplicates = get_mappings_without_duplicates(extended_mappings)

    if verbose:
        print("all mappings without duplicates:")
        for mapping in mappings_no_duplicates:
            print(mapping)

    cache = beam_search(min(len(mappings_no_duplicates), beam), mappings_no_duplicates)
    if cache:
        best_solution, best_score = cache[0][0], cache[0][1]
        solution = [mappings_no_duplicates[mapping_id] for mapping_id in best_solution]
        solution = sorted(solution, key=lambda t: t[::-1], reverse=True)
        plot_bipartite_graph(solution[:num_mappings_to_show])

    # mappings_after_coref = get_mappings_solution_after_coref(solution)
    #
    # if verbose:
    #     print()
    #     print("mappings after coreference: ")
    #     for mapping in mappings_after_coref:
    #         print(mapping)
    #
    # plot_bipartite_graph(mappings_after_coref)


def get_mappings_solution_after_coref(solution):
    mappings_after_coref = []
    for mapping in solution:
        text1_cluster, text2_cluster, similar_questions, score = mapping
        t1_cluster_entities, t2_cluster_entities = text1_cluster[1], text2_cluster[1]
        coref_t1_cluster_entities, coref_t2_cluster_entities = set(), set()

        for entity in t1_cluster_entities:
            found = False
            for key, val in text1_coref_clusters.items():
                if val == entity:
                    coref_t1_cluster_entities.add(key)
                    found = True
                    continue
            if not found:
                coref_t1_cluster_entities.add(entity)

        for entity in t2_cluster_entities:
            found = False
            for key, val in text2_coref_clusters.items():
                if val == entity:
                    coref_t2_cluster_entities.add(key)
                    found = True
                    continue
            if not found:
                coref_t2_cluster_entities.add(entity)

        mappings_after_coref.append((coref_t1_cluster_entities, coref_t2_cluster_entities, similar_questions, score))
    return mappings_after_coref


def get_mappings_without_duplicates(extended_mappings):
    seen_tuples = {}
    mappings_no_duplicates = []
    for mapping in extended_mappings:
        if (mapping[0][0], mapping[1][0]) in seen_tuples:
            continue
        mappings_no_duplicates.append(mapping)
        seen_tuples[(mapping[0][0], mapping[1][0])] = True
    return mappings_no_duplicates



def get_extended_mappings_from_clusters_scores(clusters_scores, text1_clusters_of_questions,
                                               text2_clusters_of_questions, text1_clusters_of_entities,
                                               text2_clusters_of_entities):
    extended_mappings = []
    for map in clusters_scores:
        cluster_base, cluster_target, similar_questions, score = map[0], map[1], map[2], map[3]
        if score < entities_mapping_final_threshold:
            continue
        extended_mappings.append((cluster_base, cluster_target, similar_questions, score))
        base_connected_clusters = get_connected_clusters(text1_clusters_of_questions, cluster_base[0])
        target_connected_clusters = get_connected_clusters(text2_clusters_of_questions, cluster_target[0])

        if verbose:
            print("enitites mapping: ", cluster_base, cluster_target, similar_questions, score)
            print()
            print("more connections between base and target:")

        curr_new_mappings = []
        for tup_connected_base in base_connected_clusters:
            verb_connected_base, cluster_connected_base, side_connected_base = tup_connected_base[1], \
                                                                               text1_clusters_of_entities[
                                                                                   tup_connected_base[0] - 1], \
                                                                               tup_connected_base[2]
            for tup_connected_target in target_connected_clusters:
                verb_connected_target, cluster_connected_target, side_connected_target = tup_connected_target[1], \
                                                                                         text2_clusters_of_entities[
                                                                                             tup_connected_target[
                                                                                                 0] - 1], \
                                                                                         tup_connected_target[2]
                if side_connected_base != side_connected_target:
                    continue

                if are_verbs_in_similar_questions(verb_connected_base, verb_connected_target, similar_questions):
                    added_maps, added_score = get_mapping_score(clusters_scores, cluster_connected_base,
                                                                cluster_connected_target)
                    extended_mappings.append((cluster_connected_base, cluster_connected_target,
                                              similar_questions + added_maps, score + added_score))
                    curr_new_mappings.append((cluster_connected_base, cluster_connected_target,
                                              similar_questions + added_maps, score + added_score))

                    if verbose:
                        if side_connected_base == 'R':
                            print(cluster_base, verb_connected_base, cluster_connected_base)
                            print(cluster_target, verb_connected_target, cluster_connected_target)
                        else:
                            print(cluster_connected_base, verb_connected_base, cluster_base)
                            print(cluster_connected_target, verb_connected_target, cluster_target)

        if verbose:
            print("more mappings:")
            for new_map in curr_new_mappings:
                print(new_map)
            print()
    return extended_mappings


def print_corpus_entities(text1_corpus_entities, text2_corpus_entities):
    print("text1 entities:")
    for entity in text1_corpus_entities:
        print(entity)
    print()
    print("text2 entities:")
    for entity in text2_corpus_entities:
        print(entity)
    print()

def print_clusters_of_questions(text1_clusters_of_questions, text2_clusters_of_questions):
    print("text1 clusters of questions: ")
    for question in text1_clusters_of_questions:
        print(question)
    print()
    print("text2 clusters of questions: ")
    for question in text2_clusters_of_questions:
        print(question)
    print()


def print_clusters_of_entities(text1_clusters_of_entities, text2_clusters_of_entities):
    print("text1 clusters of entities: ")
    for entity in text1_clusters_of_entities:
        print(entity)
    print()
    print("text2 clusters of entities: ")
    for entity in text2_clusters_of_entities:
        print(entity)
    print()


def get_total_score_consistent_mappings(mapping_indices, mappings_no_dups):
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
  cache = []
  if len(mappings_no_dups) >= B:
    cache = [({i}, mappings_no_dups[i][-1]) for i in range(B)]
    while True:
      start_cache = cache.copy()
      for tup in cache:
        mapping_indices = tup[0]
        for j in range(len(mappings_no_dups)):
          if j in mapping_indices:
            continue
          new_mapping_indices, curr_score = get_total_score_consistent_mappings(mapping_indices.union({j}), mappings_no_dups)
          if (new_mapping_indices, curr_score) not in cache:
            cache.append((new_mapping_indices, curr_score))
      cache = sorted(cache, key=lambda t: t[1], reverse=True)
      cache = cache[:B]
      if cache == start_cache:
        break
  return cache

def get_connected_clusters(text_clusters_of_questions, text_cluster_entities_idx):
    i = text_cluster_entities_idx
    curr_cluster = text_clusters_of_questions[i - 1][1:]
    connected_clusters_ids = set()
    for set_of_questions in curr_cluster:
        for question_record in set_of_questions:
            _, _, verb, side, timestep = question_record
            for j, set_of_other_questions in text_clusters_of_questions:
                for other_question_record in set_of_other_questions:
                    _, _, other_verb, other_side, other_timestep = other_question_record
                    if other_verb == verb and other_timestep == timestep and other_side != side:
                        connected_clusters_ids.add((j, other_verb, other_side))
    return connected_clusters_ids


def are_verbs_in_similar_questions(verb_connected_base, verb_connected_target, similar_questions):
    for tup in similar_questions:
        v1, v2 = tup[0][2], tup[1][2]
        if verb_connected_base == v1 and verb_connected_target == v2:
            return True
    return False


def get_mapping_score(clusters_scores, cluster1, cluster2):
  for map in clusters_scores:
    if map[0][0] == cluster1[0] and map[1][0] == cluster2[0]:
      return map[-2], map[-1]
  return [], 0






def get_cosine_sim_between_sentences(sentences1, sentences2):
    result = []

    embeddings1 = questions_similarity_embedder.encode([tup[0] for tup in sentences1], convert_to_tensor=True)
    embeddings2 = questions_similarity_embedder.encode([tup[0] for tup in sentences2], convert_to_tensor=True)
    cosine_scores = util.pytorch_cos_sim(embeddings1, embeddings2)
    # Output the pairs with their score
    for i in range(len(sentences1)):
        for j in range(len(sentences2)):
            result.append((cosine_scores[i][j], sentences1[i], sentences2[j]))
    result.sort(key=lambda x: x[0], reverse=True)
    return result


def get_entities_similarity_score(sentBert_similarity_map, questions1, questions2, cos_sim_threshold):
    total_score = 0
    similar_questions = []
    for i in range(len(questions1)):
        for j in range(len(questions2)):
            if (questions1[i], questions2[j]) in sentBert_similarity_map:
                curr_score = sentBert_similarity_map[(questions1[i], questions2[j])]
                if curr_score >= cos_sim_threshold:
                    similar_questions.append((questions1[i], questions2[j]))
                    total_score += curr_score
    return round(total_score, 3), similar_questions


def get_questions_list_from_cluster(cluster_of_questions):
    return [q for tup in cluster_of_questions for q in tup[1]]


def should_filter_questions(tup):
    q1, qq_subject_verb_obj, q2, q2_subject_verb_obj = tup[1][0], tup[1][1], tup[2][0], tup[2][1]
    if qq_subject_verb_obj != q2_subject_verb_obj:
        return True
    q1_list, q2_list = q1.split(' '), q2.split(' ')
    if (q1_list[0] == 'where' and q1_list[1] != 'where') or (q1_list[0] != 'where' and q2_list[0] == 'where'):
        return True
    return False


def get_sent_bert_similarity_map_between_questions(questions1, questions2):
    sent_bert_similarity_map = {}
    cos_sim_all_questions_result = get_cosine_sim_between_sentences(questions1, questions2)
    for tup in cos_sim_all_questions_result:
        if should_filter_questions(tup):
            continue
        cos_sim = round(tup[0].tolist(), 3)
        sent_bert_similarity_map[(tup[1], tup[2])] = cos_sim
    return sent_bert_similarity_map


def get_entities_clusters_scores(text1_clusters_of_entities, text1_clusters_of_questions, text2_clusters_of_entities,
                                 text2_clusters_of_questions):
    questions1 = get_questions_list_from_cluster(text1_clusters_of_questions)
    questions2 = get_questions_list_from_cluster(text2_clusters_of_questions)
    sent_bert_similarity_map = get_sent_bert_similarity_map_between_questions(questions1, questions2)

    clusters_scores = []
    total_count, pass_threshold_count = 0, 0

    for i in range(len(text1_clusters_of_questions)):
        for j in range(len(text2_clusters_of_questions)):
            questions1 = list(text1_clusters_of_questions[i][1])
            questions2 = list(text2_clusters_of_questions[j][1])
            curr_score, similar_questions = get_entities_similarity_score(sent_bert_similarity_map, questions1,
                                                                          questions2, cos_sim_threshold)
            total_count += 1
            if curr_score > 0:
                pass_threshold_count += 1
                clusters_scores.append(((i + 1, text1_clusters_of_entities[i][1]),
                                        (j + 1, text2_clusters_of_entities[j][1]), similar_questions, curr_score))
    if verbose:
        print("number of clusters in text1: " + str(len(text1_clusters_of_questions)))
        print("number of clusters in text2: " + str(len(text2_clusters_of_questions)))
        print("number of pairs of clusters above cosine similarity threshold: " + str(pass_threshold_count))
        print("out of total count of pairs: " + str(total_count))
    return clusters_scores


def convert_cluster_set_to_string(cluster_set):
    cluster_set = str(cluster_set)
    cluster_set = cluster_set[1:-1]
    cluster_set = cluster_set.split(',')
    cluster_set = "\n".join(cluster_set)
    return cluster_set


def plot_bipartite_graph(clusters_scores):
    B = nx.Graph()
    left_vertices = []

    for i, quadruple in enumerate(clusters_scores):
        left, _, similar_questions, score = quadruple
        left_vertices.append(convert_cluster_set_to_string(left))

    right_vertices = []
    for i, quadruple in enumerate(clusters_scores):
        _, right, similar_questions, score = quadruple
        right_vertices.append(convert_cluster_set_to_string(right))

    B.add_nodes_from(left_vertices, bipartite=0)
    B.add_nodes_from(right_vertices, bipartite=1)

    for i, quadruple in enumerate(clusters_scores):
        left, right, similar_questions, score = quadruple
        B.add_edge(convert_cluster_set_to_string(left), convert_cluster_set_to_string(right), weight=score)

    plt.figure(figsize=(24, 8))

    top_nodes = [n for n in B.nodes if B.nodes[n]['bipartite'] == 0]

    pos = nx.bipartite_layout(B, top_nodes)
    nx.draw_networkx_nodes(B, pos, node_size=50)
    nx.draw_networkx_labels(B, pos, font_size=1)
    labels = nx.get_edge_attributes(B, 'weight')
    nx.draw_networkx_edge_labels(B, pos, label_pos=0.25, edge_labels=labels)
    nx.draw(B, pos=pos, with_labels=True)
    plt.show()

    # matching = nx.max_weight_matching(B, maxcardinality=True)
    # print("Maximum Weight Matches: ")
    # for match in matching:
    #     left, right = match[0], match[1]
    #     if left in left_vertices:
    #         print(left, right)
    #     else:
    #         print(right, left)

def get_clustering_result(answer_question_map, corpus_entities, distance_threshold):
    filtered_corpus_entities = []
    filtered_answer_question_map = {}
    for entity in corpus_entities:
        if len(entity.split(' ')) <= max_num_words_in_entity:
            filtered_answer_question_map[entity] = answer_question_map[entity]
            filtered_corpus_entities.append(entity)

    corpus_entities, answer_question_map = filtered_corpus_entities, filtered_answer_question_map

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


def get_clusters_of_questions(clusters_of_entities, answer_question_map):
    clusters_of_questions = []
    for tup in clusters_of_entities:
        idx = tup[0]
        entities = tup[1]
        questions = []
        for entity in entities:
            list_of_questions = answer_question_map[entity]
            for q in list_of_questions:
                questions.append(q)
        clusters_of_questions.append((idx, set(questions)))
    return clusters_of_questions





