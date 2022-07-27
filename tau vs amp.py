from eztao.carma import gp_psd
from eztao.ts import drw_fit
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import sqlite3
from scipy import interpolate
from eztao.carma import DRW_term
from eztao.ts import gpSimByTime

band_list = ['u','g','r','i','z','y']

def make_AGN_model(t, tau, amp):
    DRW_kernel = DRW_term(amp, tau)
    t, y, yerr = gpSimByTime(DRW_kernel, 1000, t-np.min(t), factor=10, nLC=1, log_flux=True)
    return y + 22., yerr

def convert(s):
    new = ""
    for x in s:
        new += x
    num = float(new)
    return num


Tau = np.zeros(100)
Amp = np.zeros(100)

for x in range(100):
    with open("/Users/colevogt/Downloads/PennSROP/AGN.txt", 'r') as file:
        data = file.readlines()
        tauStr = [data[2 * x][i] for i in range(7)]
        TempTau = convert(tauStr)
        if (TempTau < 4):
            tau = TempTau
            ampStr = [data[2 * x + 1][j] for j in range(9)]
            amp = convert(ampStr)
    file.close
    Tau[x] = tau
    Amp[x] = amp
plt.xlabel('Tau [log(days)]')
plt.ylabel('Amp [log(mag)]')
plt.scatter(Tau, Amp)
plt.show()