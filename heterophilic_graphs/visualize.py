import matplotlib.pyplot as plt
import numpy as np


layers = np.array([2,4,8,16,32,64,128])
rhythm = np.array([0.788,0.795,0.757,0.788,0.795,0.817,0.811])
G2 = np.array([0.768,0.751,0.755,0.741,0.78,0.786,0.775])
dropout = np.array([0.812,0.805,0.807,0.75,0.349,0.526,0.316])


plt.plot(layers,rhythm,color='r',label='GraphRhythm-GCN')
plt.plot(layers,G2,color='b',label='G2-GCN')
plt.plot(layers,dropout,color='g',label='Dropout-GCN')
plt.xlabel("Number of Layers")
plt.ylabel("Accuracy")
plt.legend(fontsize=15)
plt.show()
plt.savefig("cora_plot.png")
