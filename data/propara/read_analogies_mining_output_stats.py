import numpy as np
import xlrd
import pandas as pd
import scipy.stats as stats

import matplotlib.pyplot as plt
pairs = 'ProParaPairs.xlsx'
fmq_top_sheet_name = 'FMQ_sim_07_pair_1_100'
fmq_quarter1_sheet_name = 'FMQ_sim_07_pair_18964_19003'
fmq_quarter2_sheet_name = 'FMQ_sim_07_pair_37928_37967'
fmq_quarter3_sheet_name = 'FMQ_sim_07_pair_56892_56931'
fmq_bottom_sheet_name   = 'FMQ_sim_07_pair_75817_75856'

fmv_top_sheet_name = 'FMV_sim_05_pair_1_100'
fmv_quarter1_sheet_name = 'FMV_sim_05_pair_18964_19003'
fmv_quarter2_sheet_name = 'FMV_sim_05_pair_37928_37967'
fmv_quarter3_sheet_name = 'FMV_sim_05_pair_56892_56931'
fmv_bottom_sheet_name   = 'FMV_sim_05_pair_75817_75856'

sbert_top_sheet_name = 'SBERT_pair_1_100'
sbert_quarter1_sheet_name = 'SBERT_pair_18964_19003'
sbert_quarter2_sheet_name = 'SBERT_pair_37928_37967'
sbert_quarter3_sheet_name = 'SBERT_pair_56892_56931'
sbert_bottom_sheet_name   = 'SBERT_pair_75817_75856'

random_pairs = '100_random_pairs'

k = 100

top_range = [1, 101]
quarter1_range = [18964, 19003]
quarter2_range = [37928, 37967]
quarter3_range = [56892, 56931]
bottom_range = [75817, 75856]

gains = {'Not': 0, 'Sub': 1, 'Self': 2, 'Close': 3, 'Far': 4}

def get_tuples(unique_four_lists, sheets):
    workbook = xlrd.open_workbook(pairs)
    lists = {'top': [], 'q1': [], 'q2': [], 'q3': [], 'bottom': []}
    top, q1, q2, q3, bottom = sheets
    worksheet = workbook.sheet_by_name(top)
    for i in range(1, 101):
        key = worksheet.cell_value(i, 0) + worksheet.cell_value(i, 1)
        lists['top'].append(key)
        if key not in unique_four_lists:
            unique_four_lists[key] = True

    worksheet = workbook.sheet_by_name(q1)
    for i in range(1, 41):
        key = worksheet.cell_value(i, 0) + worksheet.cell_value(i, 1)
        lists['q1'].append(key)
        if key not in unique_four_lists:
            unique_four_lists[key] = True

    worksheet = workbook.sheet_by_name(q2)
    for i in range(1, 41):
        key = worksheet.cell_value(i, 0) + worksheet.cell_value(i, 1)
        lists['q2'].append(key)
        if key not in unique_four_lists:
            unique_four_lists[key] = True


    worksheet = workbook.sheet_by_name(q3)
    for i in range(1, 41):
        key = worksheet.cell_value(i, 0) + worksheet.cell_value(i, 1)
        lists['q3'].append(key)
        if key not in unique_four_lists:
            unique_four_lists[key] = True

    worksheet = workbook.sheet_by_name(bottom)
    for i in range(1, 41):
        key = worksheet.cell_value(i, 0) + worksheet.cell_value(i, 1)
        lists['bottom'].append(key)
        if key not in unique_four_lists:
            unique_four_lists[key] = True

    return lists


def intersection(lst1, lst2):
    return set(lst1).intersection(lst2)



def calc_intersections():
    unique_four_lists = {}
    fmq_pairs = get_tuples(unique_four_lists, [fmq_top_sheet_name, fmq_quarter1_sheet_name, fmq_quarter2_sheet_name, fmq_quarter3_sheet_name, fmq_bottom_sheet_name])
    fmv_pairs = get_tuples(unique_four_lists, [fmv_top_sheet_name, fmv_quarter1_sheet_name, fmv_quarter2_sheet_name, fmv_quarter3_sheet_name, fmv_bottom_sheet_name])
    sbert_pairs = get_tuples(unique_four_lists, [sbert_top_sheet_name, sbert_quarter1_sheet_name, sbert_quarter2_sheet_name, sbert_quarter3_sheet_name, sbert_bottom_sheet_name])

    workbook = xlrd.open_workbook(pairs)
    worksheet = workbook.sheet_by_name(random_pairs)
    for i in range(1, 100):
        key = worksheet.cell_value(i, 0) + worksheet.cell_value(i, 1)
        if key not in unique_four_lists:
            unique_four_lists[key] = True

    print(1)


    fmq_fmv_top_intersection = intersection(fmq_pairs['top'], fmv_pairs['top'])
    fmq_fmv_q1_intersection = intersection(fmq_pairs['q1'], fmv_pairs['q1'])
    fmq_fmv_q2_intersection = intersection(fmq_pairs['q2'], fmv_pairs['q2'])
    fmq_fmv_q3_intersection = intersection(fmq_pairs['q3'], fmv_pairs['q3'])
    fmq_fmv_bottom_intersection = intersection(fmq_pairs['bottom'], fmv_pairs['bottom'])

    fmq_sbert_top_intersection = intersection(fmq_pairs['top'], sbert_pairs['top'])
    fmq_sbert_q1_intersection = intersection(fmq_pairs['q1'], sbert_pairs['q1'])
    fmq_sbert_q2_intersection = intersection(fmq_pairs['q2'], sbert_pairs['q2'])
    fmq_sbert_q3_intersection = intersection(fmq_pairs['q3'], sbert_pairs['q3'])
    fmq_sbert_bottom_intersection = intersection(fmq_pairs['bottom'], sbert_pairs['bottom'])

    fmv_sbert_top_intersection = intersection(fmv_pairs['top'], sbert_pairs['top'])
    fmv_sbert_q1_intersection = intersection(fmv_pairs['q1'], sbert_pairs['q1'])
    fmv_sbert_q2_intersection = intersection(fmv_pairs['q2'], sbert_pairs['q2'])
    fmv_sbert_q3_intersection = intersection(fmv_pairs['q3'], sbert_pairs['q3'])
    fmv_sbert_bottom_intersection = intersection(fmv_pairs['bottom'], sbert_pairs['bottom'])


def metrics_k(tuples):
    precisions = []
    AP = []
    CG = []
    DCG = []
    IDCG = []
    NDCG = []

    for i in range(1, k + 1):
        TP_seen = []
        gain_i = 0
        discount_gain_i = 0
        count_true = 0
        idcg_i = 0
        for j in range(i):
            label = tuples[j][1]
            if label == 1.0:
                count_true += 1
                TP_seen.append(count_true)
            else:
                TP_seen.append(0)
            gain_i += gains[tuples[j][2]]
            discount_gain_i += gains[tuples[j][2]] / np.log2(j+2)
            idcg_i += gains['Far'] / np.log2(j+2)
        CG.append(gain_i)
        DCG.append(discount_gain_i)
        IDCG.append(idcg_i)
        ndcg_i = discount_gain_i / idcg_i
        NDCG.append(ndcg_i)
        N_k = min(k, count_true)
        AP_i = 1 / N_k * sum([x / i for x in TP_seen])
        AP.append(AP_i)
        precision_i = round(count_true / i, 3)
        precisions.append(precision_i)
        if i % 25 == 0:
            print("metrics: " + str(i)  +  ": " + str(round(precision_i,2)) + "," + str(round(AP_i, 2)) + "," + str(round(ndcg_i, 2)))

    return precisions, AP, NDCG


def convert_label_num_to_desc(labels):
    temp = []
    for l in labels:
        if l == 0.0:
            temp.append('Far')
        elif l == 1.0:
            temp.append('Close')
        elif l == 2.0:
            temp.append('Self')
        elif l == 3.0:
            temp.append('Sub')
        else:
            temp.append('Not')
    return temp

def read_top_100(worksheet):
    top_scores, top_labels, top_labels_detailed = [], [], []
    for i in range(1, 101):
        top_scores.append(worksheet.cell_value(i, 2))
        top_labels.append(worksheet.cell_value(i, 3))
        top_labels_detailed.append(worksheet.cell_value(i, 4))
    return top_scores, top_labels, top_labels_detailed


def compare_fmv_fmq_analogies_mining():
    workbook = xlrd.open_workbook(pairs)
    fmq_top_scores, fmq_top_labels, fmq_top_labels_detailed = read_top_100(workbook.sheet_by_name(fmq_top_sheet_name))
    fmv_top_scores, fmv_top_labels, fmv_top_labels_detailed = read_top_100(workbook.sheet_by_name(fmv_top_sheet_name))

    fmv_top_labels_detailed = convert_label_num_to_desc(fmv_top_labels_detailed)
    fmq_top_labels_detailed = convert_label_num_to_desc(fmq_top_labels_detailed)

    ndcg = {}
    precision = {}
    ap = {}

    print("FMV")
    tuples = list(zip(fmv_top_scores, fmv_top_labels, fmv_top_labels_detailed))
    metrics_k_results = metrics_k(tuples)
    precision['FMV'] = metrics_k_results[0]
    ap['FMV'] = metrics_k_results[1]
    ndcg['FMV'] = metrics_k_results[2]

    # fmv_mAP = (1 / 100) * sum(methods_AP['FMV'])

    fmq_metrics = {}
    print("FMQ")
    tuples = list(zip(fmq_top_scores, fmq_top_labels, fmq_top_labels_detailed))
    metrics_k_results = metrics_k(tuples)

    precision['FMQ'] = metrics_k_results[0]
    ap['FMQ'] = metrics_k_results[1]
    ndcg['FMQ'] = metrics_k_results[2]

    # fmq_mAP = (1 / 100) * sum(methods_AP['FMQ'])

    print(1)

    for method_name, values in ndcg.items():
        x = [i for i in range(1, k + 1)]
        y = values
        if method_name == 'FMV':
            plt.plot(x, y, label=method_name, linestyle='--')
        else:
            plt.plot(x, y, label=method_name)


        plt.xlabel('K')
        plt.ylabel('Normalized Discount Cumulative Gain (NDCG)')

    plt.ylim(0, 1.01)
    plt.xlim(1, 100)
    plt.legend()
    plt.show()


    for method_name, values in ap.items():
        x = [i for i in range(1, k + 1)]
        y = values
        if method_name == 'FMV':
            plt.plot(x, y, label=method_name, linestyle='--')
        else:
            plt.plot(x, y, label=method_name)

        plt.xlabel('K')
        plt.ylabel('Average Precision (AP)')

    plt.ylim(0, 1.01)
    plt.xlim(1, 100)
    plt.legend()
    plt.show()

    for method_name, values in precision.items():
        x = [i for i in range(1, k + 1)]
        y = values
        if method_name == 'FMV':
            plt.plot(x, y, label=method_name, linestyle='--')
        else:
            plt.plot(x, y, label=method_name)

        plt.xlabel('K')
        plt.ylabel('Precision (P)')

    plt.ylim(0, 1.01)
    plt.xlim(1, 100)
    plt.legend()
    plt.show()







if __name__ == '__main__':
    # calc_intersections()
    compare_fmv_fmq_analogies_mining()






