#%%
# import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from astropy.io import fits
import astropy.units as u
from astropy.constants import c
from matplotlib.ticker import MultipleLocator
from matplotlib.lines import Line2D
import scipy.interpolate as interp
import matplotlib.patheffects as pe
import sys
import os
print(os.getcwd())
sys.path.append("/data/ally/thesis_code")
sys.path.append("/data/ally/thesis_code/RV_plots_exocomets")

from thesis_code.plotting_params import use_my_style
use_my_style()


from corrections import *

my_path = os.getcwd() + '/'

obj_name = 'PDS_70'
instrument_name = 'HARPS'
data_path = '/data/aline/ec/' + obj_name + '/' + instrument_name + '/TELLURICS/telluric_corrected/'
fits_path = '/data/aline/ec/' + obj_name + '/' + instrument_name + '/FITS_pair/'

file_list = ['HARPS.2018-03-29T06:33:21.929',
             'HARPS.2018-03-29T06:48:53.411',
             'HARPS.2018-03-29T07:04:24.413',
             'HARPS.2018-03-29T07:19:55.404',
             'HARPS.2018-03-29T07:36:39.719',
             'HARPS.2018-03-29T07:52:10.411',
             'HARPS.2018-03-30T05:41:37.996',
             'HARPS.2018-03-30T05:57:09.427',
             'HARPS.2018-03-30T06:12:40.418',
             'HARPS.2018-03-30T08:22:46.842',
             'HARPS.2018-03-30T08:38:17.403',
             'HARPS.2018-03-30T08:53:48.465',
             'HARPS.2018-03-31T03:39:16.713',
             'HARPS.2018-03-31T03:54:47.425',
             'HARPS.2018-03-31T06:35:19.070',
             'HARPS.2018-03-31T06:50:49.413',
             'HARPS.2018-03-31T08:28:24.178',
             'HARPS.2018-03-31T08:43:55.480',
             'HARPS.2018-04-18T05:12:34.801',
             'HARPS.2018-04-19T05:04:01.822',
             'HARPS.2018-04-19T05:19:33.003',
             'HARPS.2018-04-20T05:19:03.239',
             'HARPS.2018-04-20T05:49:35.891',
             'HARPS.2018-04-21T05:51:29.174',
             'HARPS.2018-04-22T05:25:12.656',
             'HARPS.2018-04-23T04:50:54.677',
             'HARPS.2018-05-01T04:29:40.281',
             'HARPS.2018-05-01T05:00:11.202',
             'HARPS.2018-05-06T03:28:07.286',
             'HARPS.2018-05-06T03:58:38.487',
             'HARPS.2018-05-13T05:09:56.832',
             'HARPS.2018-05-13T05:40:27.782',
             'HARPS.2019-02-13T08:32:20.542',
             'HARPS.2019-02-13T09:02:50.943',
             'HARPS.2019-05-01T03:26:54.083',
             'HARPS.2019-05-01T04:07:23.608',
             'HARPS.2020-02-25T06:48:17.236',
             'HARPS.2020-02-25T07:18:48.758',
             'HARPS.2020-02-26T06:38:53.902',
             'HARPS.2020-02-26T07:09:26.084',
             'HARPS.2020-02-27T04:53:10.180',
             'HARPS.2020-02-27T05:23:41.661',
             'HARPS.2020-02-29T05:14:52.221',
             'HARPS.2020-02-29T05:45:23.028',
             'HARPS.2020-03-12T05:12:57.551',
             'HARPS.2020-03-12T06:02:39.146',
             'HARPS.2020-03-13T04:36:00.449',
             'HARPS.2020-03-13T05:06:32.211',
             'HARPS.2020-03-14T06:32:01.044',
             'HARPS.2020-03-14T07:02:32.254',
             'HARPS.2020-03-15T04:16:16.025',
             'HARPS.2020-03-15T04:46:47.164']

date_list = ['2018-03-29', '2018-03-30', '2018-03-31', '2018-04-18', '2018-04-19', '2018-04-20',
             '2018-04-21', '2018-04-22', '2018-04-23', '2018-05-01', '2018-05-06', '2018-05-13']

Na1_air, Na2_air = 588.995095, 589.592424
Na1_vac, Na2_vac = 589.1583264, 589.7558147

c_light = (c.to('km/s')).value
v_system = np.float64(6.01) # bibi

def find_RV(wave, Na_vac):
    wave = np.array(wave) * u.nm
    RV = wave.to(u.km / u.s, u.doppler_optical(Na_vac * u.nm)).value #replace wavelength.mean() with Na in vac
    #RV = (velocity + (vsys * u.km / u.s)).value
    return RV

wave1_dict, wave2_dict = {}, {}
flux1_dict, flux2_dict = {}, {}
fluxerr1_dict, fluxerr2_dict = {}, {}

time_dict = {}
for date in date_list:
    wave1_time, wave2_time = {}, {}
    flux1_time, flux2_time = {}, {}
    fluxerr1_time, fluxerr2_time = {}, {}

    time_list = []
    for file in file_list:
        if file[6:16] == date:
            time = file[-12:]
            time_list.append(time)
    time_dict[date] = time_list

    year = date[0:4]

    ### Find data
    for time in time_dict[date]:
        order = 56

        chunk1_list = [Na1_vac - 0.25, Na1_vac + 0.25]
        chunk2_list = [Na2_vac - 0.25, Na2_vac + 0.25]
        
        h5_path = data_path + instrument_name + '.' + date + 'T' + time + '_telluric_corrected.h5'

        with pd.HDFStore(h5_path, 'r') as store:
            df = store[str(order)]
            wavelength = df['wavelength'].values
            flux = df['flux'].values
            fluxerr = df['flux_err'].values
            model = df['model_tel'].values

            # Mask removed lines (interpolated when correcting tellurics)
            removed_df = pd.read_csv(fits_path + 'removed_lines.csv')
            file = 'HARPS.' + date + 'T' + time

            if file in removed_df['file'].values:
                removed_df = removed_df.set_index('file')
                wave1_min, wave1_max = removed_df.loc[file, 'wave1_min'], removed_df.loc[file, 'wave1_max']
                wave2_min, wave2_max = removed_df.loc[file, 'wave2_min'], removed_df.loc[file, 'wave2_max']

                mask = ~(((wavelength >= wave1_min) & (wavelength <= wave1_max)) | ((wavelength >= wave2_min) & (wavelength <= wave2_max)))
                wavelength, flux, fluxerr = wavelength[mask], flux[mask], fluxerr[mask]

            wavelength1_norm, flux1_norm, fluxerr1_norm, wavelength2_norm, flux2_norm, fluxerr2_norm = norm_flux_doublet(wavelength, flux, fluxerr, chunk1_list, chunk2_list)

            wave1_time[time], wave2_time[time] = wavelength1_norm, wavelength2_norm
            flux1_time[time], flux2_time[time] = flux1_norm, flux2_norm
            fluxerr1_time[time], fluxerr2_time[time] = fluxerr1_norm, fluxerr2_norm     

    wave1_dict[date], wave2_dict[date] = wave1_time, wave2_time
    flux1_dict[date], flux2_dict[date] = flux1_time, flux2_time
    fluxerr1_dict[date], fluxerr2_dict[date] = fluxerr1_time, fluxerr2_time


### Add data in pairs
wavelength1_pair, wavelength2_pair, flux1_pair, flux2_pair, fluxerr1_pair, fluxerr2_pair = {}, {}, {}, {}, {}, {}

for date in date_list:
    if len(time_dict[date]) <= 2:
        w1_date, w2_date, f1_date, f2_date, ferr1_date, ferr2_date = [], [], [], [], [], []

        for time in time_dict[date]:
            for w1 in wave1_dict[date][time]:
                w1_date.append(w1)
            for w2 in wave2_dict[date][time]:
                w2_date.append(w2)
            for f1 in flux1_dict[date][time]:
                f1_date.append(f1)
            for f2 in flux2_dict[date][time]:
                f2_date.append(f2)
            for ferr1 in fluxerr1_dict[date][time]:
                ferr1_date.append(ferr1)
            for ferr2 in fluxerr2_dict[date][time]:
                ferr2_date.append(ferr2)

        zipped1 = list(map(list, zip(w1_date, f1_date, ferr1_date)))
        zipped_sorted1 = sorted(zipped1, key=lambda x: x[0])
        zipped2 = list(map(list, zip(w2_date, f2_date, ferr2_date)))
        zipped_sorted2 = sorted(zipped2, key=lambda x: x[0])

        unzipped1 = [list(t) for t in zip(*zipped_sorted1)]
        unzipped2 = [list(t) for t in zip(*zipped_sorted2)]

        wavelength1_pair[date], wavelength2_pair[date] = unzipped1[0], unzipped2[0]
        flux1_pair[date], flux2_pair[date] = unzipped1[1], unzipped2[1]
        fluxerr1_pair[date], fluxerr2_pair[date] = unzipped1[2], unzipped2[2]

    else:
        pairs = [[a, b] for a, b in zip(time_dict[date][::2], time_dict[date][1::2])]
        if len(time_dict[date]) % 2 == 1:
            pairs.append([time_dict[date][-1]])

        for p, pair in enumerate(pairs):
            w1_pair, w2_pair, f1_pair, f2_pair, ferr1_pair, ferr2_pair = [], [], [], [], [], []

            for time in pair:
                for w1 in wave1_dict[date][time]:
                    w1_pair.append(w1)
                for w2 in wave2_dict[date][time]:
                    w2_pair.append(w2)
                for f1 in flux1_dict[date][time]:
                    f1_pair.append(f1)
                for f2 in flux2_dict[date][time]:
                    f2_pair.append(f2)
                for ferr1 in fluxerr1_dict[date][time]:
                    ferr1_pair.append(ferr1)
                for ferr2 in fluxerr2_dict[date][time]:
                    ferr2_pair.append(ferr2)

            zipped1 = list(map(list, zip(w1_pair, f1_pair, ferr1_pair)))
            zipped_sorted1 = sorted(zipped1, key=lambda x: x[0])
            zipped2 = list(map(list, zip(w2_pair, f2_pair, ferr2_pair)))
            zipped_sorted2 = sorted(zipped2, key=lambda x: x[0])

            unzipped1 = [list(t) for t in zip(*zipped_sorted1)]
            unzipped2 = [list(t) for t in zip(*zipped_sorted2)]

            wavelength1_pair[date + '_' + str(p+1)], wavelength2_pair[date + '_' + str(p+1)] = unzipped1[0], unzipped2[0]
            flux1_pair[date + '_' + str(p+1)], flux2_pair[date + '_' + str(p+1)] = unzipped1[1], unzipped2[1]
            fluxerr1_pair[date + '_' + str(p+1)], fluxerr2_pair[date + '_' + str(p+1)] = unzipped1[2], unzipped2[2]

# We created a new date list because some of the dates have more than 2 exposures.
# Therefore, we separated these nights in separate pairs of exposures
# (e.g. '2018-03-29_1', '2018-03-29_2', and '2018-03-29_3')
new_date_list = [k for k in wavelength1_pair]


### Average all spectra
wave1_avg, wave2_avg = [], []
flux1_avg, flux2_avg = [], []

for date in new_date_list:
    for w1 in wavelength1_pair[date]:
        wave1_avg.append(w1)
    for w2 in wavelength2_pair[date]:
        wave2_avg.append(w2)

wave1_avg = sorted(wave1_avg)
wave2_avg = sorted(wave2_avg)
wave1_avg = list(dict.fromkeys(wave1_avg))   ### remove duplicates
wave2_avg = list(dict.fromkeys(wave2_avg))   ### remove duplicates

flux1_interp, flux2_interp = {}, {}

for date in new_date_list:
    f1_interp = interp.interp1d(wavelength1_pair[date], flux1_pair[date], bounds_error=False)(wave1_avg)
    f2_interp = interp.interp1d(wavelength2_pair[date], flux2_pair[date], bounds_error=False)(wave2_avg)

    flux1_interp[date], flux2_interp[date] = f1_interp, f2_interp

for i1 in range(len(wave1_avg)):
    f1 = []
    for date in new_date_list:
        f1.append(flux1_interp[date][i1])
    flux1_avg.append(np.average(f1))
for i2 in range(len(wave2_avg)):
    f2 = []
    for date in new_date_list:
        f2.append(flux2_interp[date][i2])
    flux2_avg.append(np.average(f2))

### Find RV
rv1_dict, rv2_dict = {}, {}

for date in new_date_list:
    rv1_dict[date] = find_RV(wavelength1_pair[date], Na1_vac)
    rv2_dict[date] = find_RV(wavelength2_pair[date], Na2_vac)

rv1_avg = find_RV(wave1_avg, Na1_vac)
rv2_avg = find_RV(wave2_avg, Na2_vac)

Na1_rv = find_RV([Na1_vac], Na1_vac)[0]
Na2_rv = find_RV([Na2_vac], Na2_vac)[0]


#%% #PLOTTING STARTS HERE   

color1 = ["#F5B217", '#eb6f3b', '#faad7f']
color2 = ['grey', 'silver']
color_comet = 'k' #'#ebbd1c'


def offset(factor, list):
    list_off = []
    for value in list:
        value_off = value + factor
        list_off.append(value_off)
    return list_off

fig, ax = plt.subplots(1, 2, figsize=(12, 6), sharex=True, sharey=True)

lw1, lw2 = 0.5, 2.5
#text_kwargs = dict(ha='right', va='center', path_effects=[pe.withStroke(linewidth=2, foreground='w')])

ax_labels = {'x': r'Radial velocity (km/s)', 'y': r'Normalised flux'}


remove_range1, remove_range2 = [589.158, 589.188], [589.758, 589.788]

rv1_remove = find_RV(remove_range1, Na1_vac)
rv2_remove = find_RV(remove_range2, Na2_vac)



for date in new_date_list:
    rv1, rv2 = np.array(rv1_dict[date]), np.array(rv2_dict[date])
    flux1, flux2 = np.array(flux1_pair[date]), np.array(flux2_pair[date])

    ax[0].plot(rv1, flux1, color='darkgrey', alpha=0.3, lw=lw1)
    ax[1].plot(rv2, flux2, color='darkgrey', alpha=0.3, lw=lw1)

    ax[0].set_xlabel(ax_labels['x'], fontsize=18)

    ax[0].set_ylabel(ax_labels['y'], fontsize=18)

for k in [0, 1]:
    ax[k].tick_params(axis='both', pad=6)

    ax[k].set_xlim([-125, 125])
    ax[k].set_ylim([0, 1.75])

    ax[k].xaxis.set_major_locator(MultipleLocator(50))
    ax[k].xaxis.set_minor_locator(MultipleLocator(25))
    ax[k].yaxis.set_major_locator(MultipleLocator(0.5))
    ax[k].yaxis.set_minor_locator(MultipleLocator(0.25))

    ax[k].plot([0, 0], [ax[k].get_ylim()[0], ax[k].get_ylim()[1]], color='k', ls='--', lw=1, dashes=(7, 3))
    ax[k].vlines([-100, -75, -50, -25, 25, 50, 75, 100], ax[k].get_ylim()[0], ax[k].get_ylim()[1], color='grey', ls=':', lw=0.5, zorder=-10)
    ax[k].hlines([0, 0.25, 0.5, 0.75, 1, 1.25, 1.5], ax[k].get_xlim()[0], ax[k].get_xlim()[1], color='grey', ls=':', lw=0.5, zorder=-10)

    dline_k = [2, 1]
    ax[k].text(0.97, 0.93, r'Na I D$_{\rm\mathbf {' + str(dline_k[k]) + r'}}$', transform=ax[k].transAxes,  color='k', ha='right', va='center', path_effects=[pe.withStroke(linewidth=2, foreground='w')])

    #ax[1].grid(which='both', color='grey', ls=':', lw=0.5, zorder=-10)
    ax[k].set_axisbelow(True)

ax[0].text(-19, 1.2, r'disc' + '\n' + r'lines', color='k', ha='center', va='center', path_effects=[pe.withStroke(linewidth=2, foreground='w')])
ax[0].annotate('', xy=(-22, 0.8), xytext=(-22, 0.8 + 0.24), arrowprops=dict(arrowstyle=']->, widthA=0, lengthA=0.4, angleA=0', lw=1.5, color='k'))
ax[0].annotate('', xy=(-16, 0.8), xytext=(-16, 0.8 + 0.24), arrowprops=dict(arrowstyle=']->, widthA=0, lengthA=0.4, angleA=0', lw=1.5, color='k'))

ax[0].text(16, 0.55, r'accretion line', color='k', ha='left', va='center', path_effects=[pe.withStroke(linewidth=2, foreground='w')])
ax[0].annotate('', xy=(5, 0.8), xytext=(13, 0.6), arrowprops=dict(arrowstyle='->', lw=1.5, color='k'))

#ax[0].annotate('', xy=(-77, 0.32), xytext=(-77, 0.32 + 1e-7), arrowprops=dict(arrowstyle=']-, widthA=3.8, lengthA=0.4, angleA=-35', lw=1.5, color='k'))
#ax[0].annotate('', xy=(-86, 0.36), xytext=(-86, 0.36 + 1e-7), arrowprops=dict(arrowstyle=']-, widthA=3, lengthA=0, angleA=-45', lw=1.5, color='k'))

# ax[0].annotate('', xy=(-120, 0.65), xytext=(-55, 0.08), arrowprops=dict(arrowstyle='-', lw=1.5, color='k'))
# ax[0].annotate('', xy=(-119, 0.645), xytext=(-119, 0.645 + 1e-7), arrowprops=dict(arrowstyle=']-, widthA=0, lengthA=0.4, angleA=0', lw=1.5, color='k'))
# ax[0].annotate('', xy=(-55.8, 0.088), xytext=(-55.8, 0.088 + 1e-7), arrowprops=dict(arrowstyle=']-, widthA=0, lengthA=0.4, angleA=-90', lw=1.5, color='k'))
# ax[0].text(-91, 0.3, r'exocomets', rotation=-45.5, color='k', ha='center', va='center', path_effects=[pe.withStroke(linewidth=2, foreground='w')])

ax[0].plot(rv1_avg, flux1_avg, color=color1[0], lw=lw2)
ax[1].plot(rv2_avg, flux2_avg, color=color1[0], lw=lw2)

line1 = [Line2D([0], [0], color='darkgrey', lw=lw1 + 0.5)]
line2 = [Line2D([0], [0], color=color1[0], lw=lw2)]
label1 = [r'HARPS data']
label2 = [r'average']

leg_kwargs = dict(loc='right', ncol=1, labelcolor='linecolor', handlelength=0.85,
                  handletextpad=0.4, labelspacing=0.25, prop=dict(size=16, weight='bold'))

leg1 = ax[1].legend(line1, label1, bbox_to_anchor=(1.02, 0.13), **leg_kwargs)
leg2 = ax[1].legend(line2, label2, bbox_to_anchor=(1.02, 0.075), **leg_kwargs)
ax[1].add_artist(leg1)
ax[1].add_artist(leg2)

# for text in leg1.get_texts():
#     text.set_path_effects([pe.withStroke(linewidth=2, foreground='w'), pe.Normal()])
# for text in leg2.get_texts():
#     text.set_path_effects([pe.withStroke(linewidth=2, foreground='w'), pe.Normal()])

plt.suptitle('PDS 70 HARPS Radial Velocity Data (2018)', fontsize=20)
#plt.margins(0.02, 0.02)
#plt.subplots_adjust(wspace=0.05, hspace=0.06)
plt.tight_layout()
fig_name = '/data/ally/thesis_code/RV_plots_exocomets/RV_HARPS_DATA_no_exo_label'
fig.savefig(fig_name + '.png', bbox_inches='tight', dpi=300, pad_inches=0.1)
fig.savefig(fig_name + '.pdf', bbox_inches='tight', dpi=300, pad_inches=0.1)
plt.show()