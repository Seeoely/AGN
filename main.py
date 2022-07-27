import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import sqlite3
import sqlalchemy
import random
import json
from scipy import interpolate
from astropy.coordinates import SkyCoord
from scipy.integrate import simps
from astropy.cosmology import WMAP9 as cosmo
from eztao.carma import DRW_term
from eztao.ts import gpSimRand, gpSimByTime
amp = 0.2
tau = 100
DRW_kernel = DRW_term(np.log(amp), np.log(tau))
t, y, yerr = gpSimRand(DRW_kernel, 10, 365*10, 200)
fig, ax = plt.subplots(1, 1, dpi=150, figsize=(8, 3))
ax.errorbar(t, y, yerr, fmt='.')
plt.xlabel('MJD')
plt.ylabel('Flux [Arbitrary Units]')
plt.tight_layout()
plt.show()