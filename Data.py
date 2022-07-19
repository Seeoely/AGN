from eztao.carma import gp_psd
from eztao.ts import drw_fit
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import sqlite3
from scipy import interpolate
from eztao.carma import DRW_term
from eztao.ts import gpSimRand, gpSimByTime

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

# This just defined the different wavelengths which LSST observes at
band_wvs = 1. / (0.0001 * np.asarray([3751.36, 4741.64, 6173.23, 7501.62, 8679.19, 9711.53]))


for x in range(120):
    with open("/Users/colevogt/Downloads/PennSROP/AGN.txt", 'r') as file:
        data = file.readlines()
        tauStr = [data[2 * x][i] for i in range(7)]
        TempTau = convert(tauStr)
        if (TempTau < 4):
            tau = TempTau
            ampStr = [data[2 * x + 1][j] for j in range(9)]
            amp = convert(ampStr)
    file.close
    def inject_agn():
        # This sets up equations we need to calculate uncertainties
        function_list = np.asarray([])
        filter_list = np.asarray([])
        for band in band_list:
            blah = np.loadtxt('/Users/colevogt/Downloads/PennSROP/filters/LSST_LSST.'+band+'.dat')
            function_list = np.append(function_list, np.trapz(blah[:, 1] / blah[:, 0], blah[:, 0]))
            filter_list = np.append(filter_list, interpolate.interp1d(blah[:, 0], blah[:, 1], bounds_error=False, fill_value=0.0))
        func_dict = {}
        bands_and_func = zip(band_list, function_list)
        for band, func in bands_and_func:
            func_dict[band] = func
        filt_dict = {}
        bands_and_filts = zip(band_list, filter_list)
        for band, func in bands_and_filts:
            filt_dict[band] = func

        # This uses a database of LSST simulated observations
        conn = sqlite3.connect("/Users/colevogt/Downloads/PennSROP/baseline_nexp1_v1.7_10yrs.db")

        # Now in order to read in pandas dataframe we need to know table name
        cursor = conn.cursor()
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
        df = pd.read_sql_query(
            'SELECT fieldRA, fieldDec, seeingFwhmEff, observationStartMJD, filter, fiveSigmaDepth, skyBrightness  FROM SummaryAllProps',conn)
        conn.close()
        # Pick a random location on the sky
        ra = np.random.uniform(0, 360)
        dec = np.random.uniform(-90, 0)

        # See if LSST is pointing at this location:
        new_db = df.where((np.abs(df['fieldRA'] - ra) < 1.75) & \
            (np.abs(df['fieldDec'] - dec) < 1.75)).dropna()
        if len(new_db) == 0:
            print('LSST was not looking here...')
            sys.exit()
        for j, myband in enumerate(band_list):
            lsst_mags = np.zeros(len(new_db))
            gind2 = np.where(new_db['filter'] == myband)
            if len(new_db['observationStartMJD'].where(new_db['filter'] == myband).dropna().values) > 1:
                new_model_mags, yerr = make_AGN_model(new_db['observationStartMJD'].where(new_db['filter'] == myband).dropna().values, tau, amp)
                lsst_mags[gind2] = new_model_mags
                if (len(lsst_mags[gind2]) > 1 & len(yerr) > 1):
                    best_fit = drw_fit(new_db['observationStartMJD'].where(new_db['filter'] == myband).dropna().values, lsst_mags[gind2], yerr)
                    best_psd = gp_psd(DRW_term(*np.log(best_fit)))
                    DRW_kernel = DRW_term(amp, tau)
                    true_psd = gp_psd(DRW_kernel)

        # now lets add noise to the LC...this involves eqns..
        g = 2.2
        h = 6.626e-27
        expTime = 30.0
        my_integrals = 10. ** (-0.4 * (lsst_mags + 48.6)) * [func_dict.get(key) for key in new_db['filter'].values]
        C = expTime * np.pi * 321.15 ** 2 / g / h * my_integrals
        fwhmeff = new_db['seeingFwhmEff'].values
        pixscale = 0.2  # ''/pixel
        neff = 2.266 * (fwhmeff / pixscale) ** 2
        sig_in = 12.7
        neff = 2.266 * (new_db['seeingFwhmEff'].values / pixscale) ** 2
        my_integrals = 10. ** (-0.4 * (new_db['skyBrightness'].values + 48.6)) * [func_dict.get(key) for key in new_db['filter'].values]
        B = expTime * np.pi * 321.15 ** 2 / g / h * my_integrals * (pixscale) ** 2
        def mag_to_flux(mag):
            return 10. ** (-0.4 * (mag + 48.6))
        snr = C / np.sqrt(C / g + (B / g + sig_in ** 2) * neff)
        err = 1.09 / snr
        mag = lsst_mags + np.random.normal(loc=0 * err, scale=err)

        return new_db['observationStartMJD'].values, mag, new_db['filter'].values
    t, mag, filters= inject_agn()

    color_dict = {'u': 'purple', 'g': 'green', 'r': 'red', 'i': 'goldenrod', 'z': 'black', 'y': 'yellow'}
    n = str(x)
    print(n)
    np.savez('AGN'+n+'.npz', t=t, mag=mag, tau=tau, amp=amp, filter=filter)