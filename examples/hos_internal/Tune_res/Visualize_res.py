
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import pandas as pd

sns.set(style="whitegrid", font_scale=2, rc={'figure.figsize':(7,6.5)}) #Change Figure/Font Size
plt.rcParams["font.family"] = 'Times New Roman'

# Coloring for blindess
# https://davidmathlogic.com/colorblind/#%23D81B60-%231E88E5-%23FFC107-%23004D40-%23c56963-%2329315f-%238ab335
coloring = ['#D81B60', "#1E88E5", "#FFC107", "#004D40", "#C56963", "#29315F", "#8AB335",
            '#FE8250', '#702865', '#EF1C70', '#7BF17C', '#BD80BD', '#5BCDAC']


# df = pd.read_csv('eval_report/Mixed-Eval_report.csv')
# df = pd.read_csv('eval_report/Missing-Eval_report.csv')
df = pd.read_csv('eval_report/Mixed-Eval_report.csv')
# print(df)
pair_fracs = []
missing = [0.1,0.2,0.3,0.4]
mismatch = [0.1,0.2,0.3,0.4]
for f in missing:
    pair_fracs.extend(list(zip([f for _ in range(4)],mismatch)))
# pair_fracs = [0.1,0.2,0.3,0.4]
pos = list(range(len(pair_fracs)))
width = 0.1

# plotting the bars
fig, ax = plt.subplots(figsize=(12,6))

# create a bar with precision data
plt.bar(pos, df['precision'], width, alpha=0.5, color='#D81B60',
        label=pair_fracs[0])

# create a bar with recall data
plt.bar([p + width for p in pos], df['recall'], width, alpha=0.5, color="#004D40",
        label=pair_fracs[1])

# create a bar with repaired_recall data
plt.bar([p + width*2 for p in pos], df['repaired_recall'], width, alpha=0.5, color="#29315F",
        label=pair_fracs[2])

# create a bar with repaired_recall data
plt.bar([p + width*3 for p in pos], df['F1'], width, alpha=0.5, color="#FFC107",
        label=pair_fracs[3])

# set the y axis label
ax.set_ylabel('Evaluation')
ax.set_xlabel('(missing, mismatch)')
ax.set_title('Evaluation Report for Tuning Mixed values')
ax.set_xticks([p+1.5*width for p in pos])

# set the labels for the x ticks
ax.set_xticklabels(pair_fracs, size=10)
plt.ylim([0, 1])

# Adding the legend and showing the plot
plt.legend(['precision', 'recall','repaired_recall', 'F1'], loc='upper left',prop={'size': 12})
plt.grid()
plt.savefig('fig/Mixed_Evaluation_report.png')
plt.show()