import csv
import seaborn as sns; sns.set()

toCSV = [{'x':1.0,'y':2.0,'label':1.0}, {'x':2.0,'y':3.0,'label':0.0}, {'x':4.0, 'y':6.0, 'label':1.0}]

with open('results.csv', 'w') as csv_file:
    writer = csv.writer(csv_file)
    writer.writerow(('x', 'y', 'label'))
    for i in range(len(toCSV)):
        writer.writerow([toCSV[i].get('x'), toCSV[i].get('y'), toCSV[i].get('label')])

import pandas as pd
import matplotlib.pyplot as plt
df  = pd.read_csv("results.csv")
# df.plot()  # plots all columns against index
sns.relplot(x="x", y="y", hue="label", data=df);
# df.plot(kind='density')  # estimate density function
plt.show()
