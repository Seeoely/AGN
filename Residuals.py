import matplotlib.pyplot as plt
import numpy as np
r = np.load("AGNParam.npz")
plt.hist(r["arr_0"], 20)
plt.show()
plt.hist(r["arr_1"], 20)
plt.show()