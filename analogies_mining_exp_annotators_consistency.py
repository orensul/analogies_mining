import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
import numpy as np


map = {'Not': 0, 'Self': 1, 'Close': 2, 'Far': 3, 'Sub': 4}
def main():
    # actual values
    set1_actual = ['Close', 'Not', 'Self', 'Far', 'Sub'] * 3
    set2_actual = ['Far', 'Self', 'Sub', 'Not', 'Close'] * 3
    set3_actual = ['Close', 'Self', 'Not', 'Far', 'Sub']
    set4_actual = ['Far', 'Self', 'Not', 'Close', 'Sub']

    actual = [map[l] for l in set1_actual] + [map[l] for l in set2_actual] + \
             [map[l] for l in set3_actual] + [map[l] for l in set4_actual]

    # predicted values
    predicted = [map[l] for l in ['Sub', 'Not', 'Self', 'Sub', 'Sub',
                 'Close', 'Not', 'Self', 'Far', 'Far',
                 'Close', 'Far', 'Self', 'Far', 'Sub',

                 'Close', 'Self', 'Far', 'Not', 'Sub',
                 'Close', 'Self', 'Close', 'Not', 'Self',
                 'Far', 'Self', 'Sub', 'Not', 'Close',

                 'Close', 'Self', 'Not', 'Sub', 'Far',

                 'Far', 'Self', 'Not', 'Close', 'Sub'

                 ]]

    print(predicted)
    print(actual)


    cm = confusion_matrix(y_true=actual, y_pred=predicted)
    print(cm)
    plot_confusion_matrix(cm, classes=['Not', 'Self', 'Close', 'Far', 'Sub'])

    # classification report for precision, recall f1-score and accuracy
    matrix = classification_report(actual, predicted)
    print('Classification report : \n', matrix)


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


