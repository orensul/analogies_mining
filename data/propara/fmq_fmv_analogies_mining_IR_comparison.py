
import numpy as np
import xlrd
import matplotlib.pyplot as plt

pairs = 'ProParaPairs.xlsx'
fmq_top_sheet_name = 'FMQ_sim_07_pair_1_100'
fmv_top_sheet_name = 'FMV_sim_05_pair_1_100'

k = 100
gains = {'Not': 0, 'Sub': 1, 'Self': 2, 'Close': 3, 'Far': 4}


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


def read_top_100(worksheet):
    top_scores, top_labels, top_labels_detailed = [], [], []
    for i in range(1, 101):
        top_scores.append(worksheet.cell_value(i, 2))
        top_labels.append(worksheet.cell_value(i, 3))
        top_labels_detailed.append(worksheet.cell_value(i, 4))
    return top_scores, top_labels, top_labels_detailed


def convert_label_num_to_desc(labels):
    label_description = []
    for l in labels:
        if l == 0.0:
            label_description.append('Far')
        elif l == 1.0:
            label_description.append('Close')
        elif l == 2.0:
            label_description.append('Self')
        elif l == 3.0:
            label_description.append('Sub')
        else:
            label_description.append('Not')
    return label_description


if __name__ == '__main__':
    compare_fmv_fmq_analogies_mining()