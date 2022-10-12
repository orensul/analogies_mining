
import xlrd

pairs = 'ProParaPairs.xlsx'

fmq_top_sheet_name = 'FMQ_sim_07_pair_1_100'
fmq_quarter1_sheet_name = 'FMQ_sim_07_pair_18964_19003'
fmq_quarter2_sheet_name = 'FMQ_sim_07_pair_37928_37967'
fmq_quarter3_sheet_name = 'FMQ_sim_07_pair_56892_56931'
fmq_bottom_sheet_name = 'FMQ_sim_07_pair_75817_75856'

fmv_top_sheet_name = 'FMV_sim_05_pair_1_100'
fmv_quarter1_sheet_name = 'FMV_sim_05_pair_18964_19003'
fmv_quarter2_sheet_name = 'FMV_sim_05_pair_37928_37967'
fmv_quarter3_sheet_name = 'FMV_sim_05_pair_56892_56931'
fmv_bottom_sheet_name = 'FMV_sim_05_pair_75817_75856'

sbert_top_sheet_name = 'SBERT_pair_1_100'
sbert_quarter1_sheet_name = 'SBERT_pair_18964_19003'
sbert_quarter2_sheet_name = 'SBERT_pair_37928_37967'
sbert_quarter3_sheet_name = 'SBERT_pair_56892_56931'
sbert_bottom_sheet_name = 'SBERT_pair_75817_75856'

random_pairs = '100_random_pairs'

k = 100
top_range = [1, 101]
quarter1_range = [18964, 19003]
quarter2_range = [37928, 37967]
quarter3_range = [56892, 56931]
bottom_range = [75817, 75856]
gains = {'Not': 0, 'Sub': 1, 'Self': 2, 'Close': 3, 'Far': 4}


def calc_intersections():
    """
    Print intersection between pairs in different part of the ranked list of experiment 1 (analogies mining), between
    the different methods: FMQ, FMV and SBERT.
    """
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

    print("intersection between FMQ and FMV pairs at the top 100: ")
    print(fmq_fmv_top_intersection)
    print("#pairs: " + str(len(fmq_fmv_top_intersection)))

    print("intersection between FMQ and FMV pairs at percentile 25%: ")
    print(fmq_fmv_q1_intersection)
    print("#pairs: " + str(len(fmq_fmv_q1_intersection)))

    print("intersection between FMQ and FMV pairs at percentile 50%: ")
    print(fmq_fmv_q2_intersection)
    print("#pairs: " + str(len(fmq_fmv_q2_intersection)))

    print("intersection between FMQ and FMV pairs at percentile 75%: ")
    print(fmq_fmv_q3_intersection)
    print("#pairs: " + str(len(fmq_fmv_q3_intersection)))

    print("intersection between FMQ and FMV pairs at the bottom: ")
    print(fmq_fmv_bottom_intersection)
    print("#pairs: " + str(len(fmq_fmv_bottom_intersection)))


    print("intersection between FMQ and SBERT pairs at the top 100: ")
    print(fmq_sbert_top_intersection)
    print("#pairs: " + str(len(fmq_sbert_top_intersection)))

    print("intersection between FMQ and SBERT pairs at percentile 25%: ")
    print(fmq_sbert_q1_intersection)
    print("#pairs: " + str(len(fmq_sbert_q1_intersection)))

    print("intersection between FMQ and SBERT pairs at percentile 50%: ")
    print(fmq_sbert_q2_intersection)
    print("#pairs: " + str(len(fmq_sbert_q2_intersection)))

    print("intersection between FMQ and SBERT pairs at percentile 75%: ")
    print(fmq_sbert_q3_intersection)
    print("#pairs: " + str(len(fmq_sbert_q3_intersection)))

    print("intersection between FMQ and SBERT pairs at the bottom: ")
    print(fmq_sbert_bottom_intersection)
    print("#pairs: " + str(len(fmq_sbert_bottom_intersection)))


    print("intersection between FMV and SBERT pairs at the top 100: ")
    print(fmv_sbert_top_intersection)
    print("#pairs: " + str(len(fmv_sbert_top_intersection)))

    print("intersection between FMV and SBERT pairs at percentile 25%: ")
    print(fmv_sbert_q1_intersection)
    print("#pairs: " + str(len(fmv_sbert_q1_intersection)))

    print("intersection between FMV and SBERT pairs at percentile 50%: ")
    print(fmv_sbert_q2_intersection)
    print("#pairs: " + str(len(fmv_sbert_q2_intersection)))

    print("intersection between FMV and SBERT pairs at percentile 75%: ")
    print(fmv_sbert_q3_intersection)
    print("#pairs: " + str(len(fmv_sbert_q3_intersection)))

    print("intersection between FMV and SBERT pairs at the bottom: ")
    print(fmv_sbert_bottom_intersection)
    print("#pairs: " + str(len(fmv_sbert_bottom_intersection)))


def intersection(lst1, lst2):
    return set(lst1).intersection(lst2)


def get_tuples(unique_four_lists, sheets):
    """
    Returns a dictionary with key (top / q1 / q2 / q3 / bottom) for the name of the part in the ranked list,
    and the value as a list of the pairs of paragraphs in this part.
    (Doing it by reading the different sheets in the xlsx pairs file).
    """
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


if __name__ == '__main__':
    calc_intersections()






