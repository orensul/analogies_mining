
from sentence_transformers import SentenceTransformer, util
from qa_srl import read_parsed_qasrl_verbs
from find_mappings import get_clusters_of_entities, print_clusters_of_entities, print_corpus_entities, \
    beam_search, get_mappings_without_duplicates
verbs_similarity_embedder = SentenceTransformer('msmarco-distilbert-base-v4')
clustering_embedder = verbs_similarity_embedder

max_num_words_in_entity = 7
score_boosting_same_sentence = 1
agglomerative_clustering_distance_threshold = 1
beam = 7
verbose = False


def generate_mappings(pair, cos_sim_threshold):
    """
    Returns top1, top2 and top3 solutions (if exists) for the pair of texts.
    This function implements the FMV baseline method, which is very similar to our method (FMQ),
    See Section 4.1 in the paper for description of this baseline.
    """
    text1_answer_verb_map = read_parsed_qasrl_verbs(pair[0])
    text2_answer_verb_map = read_parsed_qasrl_verbs(pair[1])

    text1_corpus_entities = list(text1_answer_verb_map.keys())
    text2_corpus_entities = list(text2_answer_verb_map.keys())

    if not text1_corpus_entities or not text2_corpus_entities:
        return None, None, None
    if verbose:
        print_corpus_entities(text1_corpus_entities, text2_corpus_entities)

    text1_clusters_of_entities = get_clusters_of_entities(text1_answer_verb_map, text1_corpus_entities,
                                                          distance_threshold=agglomerative_clustering_distance_threshold)

    text1_clusters_of_verbs = get_clusters_of_verbs(text1_clusters_of_entities, text1_answer_verb_map)

    text2_clusters_of_entities = get_clusters_of_entities(text2_answer_verb_map, text2_corpus_entities,
                                                          distance_threshold=agglomerative_clustering_distance_threshold)

    text2_clusters_of_verbs = get_clusters_of_verbs(text2_clusters_of_entities, text2_answer_verb_map)

    if verbose:
        print_clusters_of_entities(text1_clusters_of_entities, text2_clusters_of_entities)
        print_clusters_of_verbs(text1_clusters_of_verbs, text2_clusters_of_verbs)

    clusters_scores = get_sorted_entities_clusters_scores(text1_clusters_of_entities, text1_clusters_of_verbs,
                                                   text2_clusters_of_entities, text2_clusters_of_verbs, cos_sim_threshold)

    mappings_no_duplicates = get_mappings_without_duplicates(clusters_scores)
    max_mappings_before_beam_search = 20
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
        print("\n" + "top 1 solution: ")
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


def get_sorted_entities_clusters_scores(text1_clusters_of_entities, text1_clusters_of_verbs, text2_clusters_of_entities,
                                 text2_clusters_of_verbs, cos_sim_threshold):
    verbs1 = [v for tup in text1_clusters_of_verbs for v in tup[1]]
    verbs2 = [v for tup in text2_clusters_of_verbs for v in tup[1]]
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

    if verbose:
        print("number of clusters in text1: " + str(len(text1_clusters_of_verbs)))
        print("number of clusters in text2: " + str(len(text2_clusters_of_verbs)))
        print("number of pairs of clusters above cosine similarity threshold: " + str(pass_threshold_count))
        print("out of total count of pairs: " + str(total_count))

    clusters_scores = sorted(clusters_scores, key=lambda t: t[::-1], reverse=True)
    return clusters_scores


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


def print_clusters_of_verbs(text1_clusters_of_verbs, text2_clusters_of_verbs):
    print("text1 clusters of questions: ")
    for question in text1_clusters_of_verbs:
        print(question)
    print()
    print("text2 clusters of questions: ")
    for question in text2_clusters_of_verbs:
        print(question)
    print()

