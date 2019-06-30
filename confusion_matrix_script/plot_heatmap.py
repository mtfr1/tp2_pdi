import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

sns.set()

desc = ['Diamond Peach', 'Kiwi', 'Nectarine', 'Asterix Potato', 'Onion', 'Honneydew Melon', 'Taiti Lime', 'Spanish Pear', 'Cashew', 'Fuji Apple', 'Watermelon', 'Orange', 'Granny Smith Apple', 'Plum', 'Agata Potato']
files = ["lbp", "hist"]
for i in files:	
	filename = "input_" + i + ".csv"
	data = np.loadtxt(filename, delimiter = ';')
	plt.figure(figsize = (16,5))
	sns.heatmap(data, xticklabels=desc, yticklabels=desc, linewidths=.5, annot=True)
	plt.savefig(filename + '.pdf', bbox_inches='tight')