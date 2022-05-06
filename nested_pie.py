import matplotlib.pyplot as plt



def plot(labels1, labels2, colors1, colors2, fracs1, fracs2):
    fig1, ax1 = plt.subplots()
    ax1.pie(fracs1, labels=labels1, radius=1, autopct='%1.1f%%',  startangle=90, colors=colors1)
    fig1, ax2 = plt.subplots()
    ax2.pie(fracs2, labels=labels2, radius=1,  autopct='%1.1f%%',  startangle=90, colors=colors2)
    plt.show()


labels1 = 'Analogies',
# colors1 = ['#0000FF', '#FFA500',  '#FF0000']
colors1 = ['#0000FF']

labels2 = 'Self analogies', 'Close analogies'
# colors2 = ['#FFC0CB', '#800080', '#808080']
colors2 = ['#FFC0CB', '#800080']
fracs1 = [100]
fracs2 = [89, 11]
plot(labels1, labels2, colors1, colors2, fracs1, fracs2)