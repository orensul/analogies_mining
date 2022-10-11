import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

find_mappings_questions_first_sample_file_name = 'find_mappings_questions_cosine_sim.txt'


def read_questions_cosine_sim(filename, bins, labels):
    triplets = set()
    file_object = open(filename, 'r')
    for line in file_object:
        cos_sim, v1, v2 = float(line.split(';')[0]), line.split(';')[1], line.split(';')[2][:-1]
        if (cos_sim, v1, v2) not in triplets and (cos_sim, v2, v1) not in triplets:
            triplets.add((cos_sim, v1, v2))

    questions, scores = [], []
    for triple in triplets:
        questions.append(triple[1] + ';' + triple[2])
        scores.append(triple[0])

    df1 = {'questions': questions, 'scores': scores}
    df1 = pd.DataFrame(df1, columns=['questions', 'scores'])
    df1['binned'] = pd.cut(df1['scores'], bins, labels=labels)

    results = []
    for i in range(labels[0], int(labels[-1]) + 1):
        results.append(df1[(df1['binned'] == i)])
    print(results)


def plot_graph(x, y):
    plt.plot(x, y)
    for a, b in zip(x, y):
        plt.text(a, b, str(b))

    plt.xlabel('cosine similarity threshold')
    plt.ylabel('questions similarity accuracy')
    plt.show()


if __name__ == '__main__':
    bins = [0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9]
    labels = [1, 2, 3, 4, 5, 6, 7, 8]
    read_questions_cosine_sim(find_mappings_questions_first_sample_file_name, bins, labels)

    # values derived from ../paper_experiments_results/cosine_similarity_threshold_figure4/samples_for_questions_cosine_threshold.csv
    thresholds_x_axis = [0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9]
    acc_y_axis = [0.333, 0.467, 0.533, 0.667, 0.8, 0.8, 0.8, 0.8, 0.867]
    plot_graph(np.array(thresholds_x_axis), np.array(acc_y_axis))
