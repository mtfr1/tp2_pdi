import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

sns.set()

FILENAME = "input.csv"

data = np.loadtxt(FILENAME, delimiter = ';')

desc = ['Diamond Peach', 'Kiwi', 'Nectarine', 'Asterix Potato', 'Onion', 'Honneydew Melon', 'Taiti Lime', 'Spanish Pear', 'Cashew', 'Fuji Apple', 'Watermelon', 'Orange', 'Granny Smith Apple', 'Plum', 'Agata Potato']


plt.figure(figsize = (16,5))

sns.heatmap(data, xticklabels=desc, yticklabels=desc, linewidths=.5, annot=True)


plt.savefig(FILENAME + '.pdf', bbox_inches='tight')
plt.show()