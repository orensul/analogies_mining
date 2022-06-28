import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
import numpy as np

import seaborn as sns


def draw_confusion_matrix(actual, predicted, classes):
    cm = confusion_matrix(y_true=actual, y_pred=predicted)
    print(cm)



    plot_confusion_matrix(cm, classes)

    matrix = classification_report(actual, predicted)
    print('Classification report : \n', matrix)


def main():
    set1_actual = ['Close', 'Not', 'Self', 'Far', 'Sub'] * 3
    set2_actual = ['Far', 'Self', 'Sub', 'Not', 'Close'] * 3
    set3_actual = ['Close', 'Self', 'Not', 'Far', 'Far'] * 3
    set4_actual = ['Far', 'Self', 'Not', 'Close', 'Sub'] * 3
    set5_actual = ['Self', 'Close', 'Not', 'Sub', 'Sub'] * 3

    five_labels_map = {'Not': 0, 'Self': 1, 'Close': 2, 'Far': 3, 'Sub': 4}
    conversion_map = {'Self': 'Analogy', 'Close': 'Analogy', 'Far': 'Analogy', 'Not': 'Not', 'Sub': 'Analogy'}
    three_labels_map = {'Not': 0, 'Analogy': 1}
    actual_five_labels = set1_actual + set2_actual + set3_actual + set4_actual + set5_actual
    actual_two_labels = [conversion_map[l] for l in actual_five_labels]

    actual_classes_five_labels = [five_labels_map[l] for l in actual_five_labels]
    actual_classes_two_labels = [three_labels_map[l] for l in actual_two_labels]

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
    predicted_classes_two_labels = [three_labels_map[l] for l in predicted_two_labels]

    draw_confusion_matrix(actual_classes_five_labels, predicted_classes_five_labels, ['Not', 'Self', 'Close', 'Far', 'Sub'])
    draw_confusion_matrix(actual_classes_two_labels, predicted_classes_two_labels, ['No', 'Yes'])






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
    main()


