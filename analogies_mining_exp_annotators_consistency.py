import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
import numpy as np

import seaborn as sns
from statsmodels.stats import inter_rater as irr
from sklearn.metrics import cohen_kappa_score

five_labels_map = {'Not': 0, 'Self': 1, 'Close': 2, 'Far': 3, 'Sub': 4}
conversion_map = {'Self': 'Analogy', 'Close': 'Analogy', 'Far': 'Analogy', 'Not': 'Not', 'Sub': 'Analogy'}
two_labels_map = {'Not': 0, 'Analogy': 1}

def draw_confusion_matrix(actual, predicted, classes):
    cm = confusion_matrix(y_true=actual, y_pred=predicted)
    print(cm)



    plot_confusion_matrix(cm, classes)

    matrix = classification_report(actual, predicted)
    print('Classification report : \n', matrix)

def expert_annotators_check_1():


    expert1 = ['Self', 'Self', 'Close', 'Close', 'Far', 'Far', 'Not', 'Not', 'Sub', 'Sub']
    expert2 = ['Self', 'Self', 'Close', 'Close', 'Far', 'Far', 'Not', 'Not', 'Not', 'Sub']

    expert1_two_labels = [conversion_map[l] for l in expert1]
    expert1_two_labels = [two_labels_map[l] for l in expert1_two_labels]

    expert2_two_labels = [conversion_map[l] for l in expert2]
    expert2_two_labels = [two_labels_map[l] for l in expert2_two_labels]

    expert1_five_labels = [five_labels_map[l] for l in expert1]
    expert2_five_labels = [five_labels_map[l] for l in expert2]

    cohen_kappa_two_labels = round(cohen_kappa_score(expert1_two_labels, expert2_two_labels), 2)
    cohen_kappa_five_labels = round (cohen_kappa_score(expert1_five_labels, expert2_five_labels), 2)

    print("cohen kappa check 1 two labels: " + str(cohen_kappa_two_labels))
    print("cohen kappa check 1 five labels: " + str(cohen_kappa_five_labels))



def volunteer_annotators_check_2():
    set1_actual = ['Close', 'Not', 'Self', 'Far', 'Sub']
    set2_actual = ['Far', 'Self', 'Sub', 'Not', 'Close']
    set3_actual = ['Close', 'Self', 'Not', 'Far', 'Far']
    set4_actual = ['Far', 'Self', 'Not', 'Close', 'Sub']
    set5_actual = ['Self', 'Close', 'Not', 'Sub', 'Sub']


    actual_five_labels = set1_actual * 3 + set2_actual * 3 + set3_actual * 3 + set4_actual * 3 + set5_actual * 3
    actual_two_labels = [conversion_map[l] for l in actual_five_labels]

    actual_classes_five_labels = [five_labels_map[l] for l in actual_five_labels]
    actual_classes_two_labels = [two_labels_map[l] for l in actual_two_labels]

    predicted_five_labels = ['Sub', 'Not', 'Self', 'Sub', 'Sub',
                             'Close', 'Not', 'Self', 'Far', 'Far',
                             'Close', 'Not', 'Self', 'Far', 'Sub',

                             'Close', 'Self', 'Far', 'Not', 'Sub',
                             'Close', 'Self', 'Close', 'Not', 'Self',
                             'Far', 'Self', 'Sub', 'Not', 'Close',

                             'Close', 'Self', 'Not', 'Sub', 'Far',
                             'Sub', 'Self', 'Not', 'Far', 'Not',
                             'Far', 'Self', 'Far', 'Not', 'Far',

                             'Far', 'Self', 'Not', 'Close', 'Sub',
                             'Far', 'Self', 'Not', 'Close', 'Sub',
                             'Close', 'Self', 'Not', 'Close', 'Sub',
                             
                             'Self', 'Sub', 'Not', 'Sub', 'Sub',
                             'Self', 'Self', 'Not', 'Sub', 'Sub',
                             'Self', 'Self', 'Not', 'Far', 'Sub',

                             ]

    predicted_two_labels = [conversion_map[l] for l in predicted_five_labels]
    predicted_classes_five_labels = [five_labels_map[l] for l in predicted_five_labels]
    predicted_classes_two_labels = [two_labels_map[l] for l in predicted_two_labels]

    draw_confusion_matrix(actual_classes_five_labels, predicted_classes_five_labels, ['Not', 'Self', 'Close', 'Far', 'Sub'])
    draw_confusion_matrix(actual_classes_two_labels, predicted_classes_two_labels, ['No', 'Yes'])


    # fleiss kappa check 2 two labels
    actual_two_labels = [conversion_map[l] for l in set1_actual + set2_actual + set3_actual + set4_actual + set5_actual]
    actual_classes_two_labels = [two_labels_map[l] for l in actual_two_labels]
    annotator_pairs_annotations = convert_data_for_inter_annotator_clac(actual_classes_two_labels, predicted_classes_two_labels)
    fleiss_kappa_two_labels = calc_inter_annotator_agreement(annotator_pairs_annotations)
    print("fleiss kappa check 2 two labels: " + str(fleiss_kappa_two_labels))

    # fleiss kappa check 2 five labels
    actual_five_labels = set1_actual + set2_actual + set3_actual + set4_actual + set5_actual
    actual_classes_five_labels = [five_labels_map[l] for l in actual_five_labels]
    annotator_pairs_annotations = convert_data_for_inter_annotator_clac(actual_classes_five_labels, predicted_classes_five_labels)
    fleiss_kappa_five_labels = calc_inter_annotator_agreement(annotator_pairs_annotations)
    print("fleiss kappa check 2 five labels: " + str(fleiss_kappa_five_labels))






def convert_data_for_inter_annotator_clac(actual, predicted):
    annotator_pairs_annotations = {'expert': [], 'A1': [], 'A2': [], 'A3': []}
    annotator_pairs_annotations['expert'] = actual

    for i, label in enumerate(predicted):
        if i % 15 <= 4:
            annotator_pairs_annotations['A1'].append(label)
        elif 4 < i % 15 <= 9:
            annotator_pairs_annotations['A2'].append(label)
        else:
            annotator_pairs_annotations['A3'].append(label)

    return annotator_pairs_annotations







def calc_inter_annotator_agreement(annotator_pairs_annotations):
    fleiss_kappa_data = []
    for i in range(len(annotator_pairs_annotations['expert'])):
        expert_label_i = annotator_pairs_annotations['expert'][i]
        a1_label_i = annotator_pairs_annotations['A1'][i]
        a2_label_i = annotator_pairs_annotations['A2'][i]
        a3_label_i = annotator_pairs_annotations['A3'][i]
        fleiss_kappa_data.append([expert_label_i, a1_label_i, a2_label_i, a3_label_i])

    agg_fleiss_kappa_data, n_cats = irr.aggregate_raters(fleiss_kappa_data)
    fleiss_kappa_result = round(irr.fleiss_kappa(agg_fleiss_kappa_data, method='fleiss'), 2)
    return fleiss_kappa_result


def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    thresh = cm.max() / 2.
    import itertools
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.show()







if __name__ == '__main__':
    # expert_annotators_check_1()
    volunteer_annotators_check_2()


