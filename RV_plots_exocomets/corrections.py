import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from astropy.io import fits
from astropy.constants import c
from scipy.interpolate import interp1d
#import tayph.operations as ops

### Correct for BERV (barycentric Earth RV)
# def correct_berv(wavelength_obs, berv):
#     c_light = (c.to('km/s')).value
#     gamma = 1.0 + (berv / c_light)

#     wavelength_rest = wavelength_obs * gamma

#     return wavelength_rest


# ### Correct for BERV and Vsys
# def correct_berv_vsys(wavelength_obs, berv, vsys):
#     c_light = (c.to('km/s')).value
#     gamma = 1.0 + ((berv - vsys) / c_light)

#     wavelength_rest = wavelength_obs * gamma

#     return wavelength_rest


# ### Correct from air to vacuum
# def correct_airtovac(wave):
#     new_wave = ops.airtovac(wave)

#     return new_wave


# ### Remove sky lines
# def remove_line(line_range, wave_A, flux_A, fluxerr_A, wave_B, flux_B, fiber_list, file_B):
#     sky_flux = interp1d(wave_B, flux_B, bounds_error=False)(wave_A)

#     if np.shape(wave_A) < np.shape(wave_B):
#         sky_wave = wave_A
#     else:
#         sky_wave = wave_B

#     if file_B in fiber_list:
#         line_wave, line_flux = [], []
#         for i in range(len(line_range)):
#             iwave, iflux = [], []
#             for j, wl in enumerate(wave_A):
#                 if line_range[i][0] < wl < line_range[i][-1]:
#                     iwave.append(wl)
#                     iflux.append(sky_flux[j])
#             line_wave.append(iwave)
#             line_flux.append(iflux)

#         remove_wave = []
#         for i in range(len(line_range)):
#             for j, wl in enumerate(line_wave[i]):
#                 flux_max_index = line_flux[i].index(np.max(line_flux[i]))
#                 wave_max = line_wave[i][flux_max_index]
#                 if wave_max - 0.008 < wl < wave_max + 0.008:
#                     remove_wave.append(wl)
                
#         new_wave_A, new_flux_A, new_fluxerr_A = [], [], []
#         for j, wl in enumerate(wave_A):
#             if wl not in remove_wave:
#                 new_wave_A.append(wl)
#                 new_flux_A.append(flux_A[j])
#                 new_fluxerr_A.append(fluxerr_A[j])

#     else:
#         new_wave_A = wave_A
#         new_flux_A = flux_A - sky_flux
#         new_fluxerr_A = fluxerr_A

#     return new_wave_A, new_flux_A, new_fluxerr_A, sky_wave, sky_flux



# ### Remove outliers
# def remove_outlier(wavelength1, flux1, fluxerr1, wavelength2, flux2, fluxerr2):

#     wavelength1_rmv, flux1_rmv, fluxerr1_rmv = [], [], []
#     wavelength1_keep, flux1_keep, fluxerr1_keep = [], [], []
#     for i, wave in enumerate(wavelength1):
#         if i == 0 or i == len(wavelength1) - 1:
#             wavelength1_keep.append(wavelength1[i])
#             flux1_keep.append(flux1[i])
#             fluxerr1_keep.append(fluxerr1[i])
#         else:
#             diff_mean_flux = flux1[i] - np.mean([flux1[i-1], flux1[i], flux1[i+1]])
#             if diff_mean_flux > 0.2:
#                 wavelength1_rmv.append(wavelength1[i])
#                 flux1_rmv.append(flux1[i])
#                 fluxerr1_rmv.append(fluxerr1[i])
#             else:
#                 wavelength1_keep.append(wavelength1[i])
#                 flux1_keep.append(flux1[i])
#                 fluxerr1_keep.append(fluxerr1[i])

#     wavelength2_rmv, flux2_rmv, fluxerr2_rmv = [], [], []
#     wavelength2_keep, flux2_keep, fluxerr2_keep = [], [], []
#     for i, wave in enumerate(wavelength2):
#         if i == 0 or i == len(wavelength2) - 1:
#             wavelength2_keep.append(wavelength2[i])
#             flux2_keep.append(flux2[i])
#             fluxerr2_keep.append(fluxerr2[i])
#         else:
#             diff_mean_flux = flux2[i] - np.mean([flux2[i-1], flux2[i], flux2[i+1]])
#             if diff_mean_flux > 0.2:
#                 wavelength2_rmv.append(wavelength2[i])
#                 flux2_rmv.append(flux2[i])
#                 fluxerr2_rmv.append(fluxerr2[i])
#             else:
#                 wavelength2_keep.append(wavelength2[i])
#                 flux2_keep.append(flux2[i])
#                 fluxerr2_keep.append(fluxerr2[i])
    
#     list_rmv = [wavelength1_rmv, flux1_rmv, fluxerr1_rmv, wavelength2_rmv, flux2_rmv, fluxerr2_rmv]
#     list_keep = [wavelength1_keep, flux1_keep, fluxerr1_keep, wavelength2_keep, flux2_keep, fluxerr2_keep]

#     return list_rmv, list_keep



# ### Normalize a chunk of spectrum
# def norm_flux(wave, flux, chunk_list):
#     diff0_1, diff1_1 = np.absolute(np.array(wave) - chunk_list[0]), np.absolute(np.array(wave) - chunk_list[-1])
#     w1_index = [diff0_1.argmin(), diff1_1.argmin()]
#     w1_norm = wave[w1_index[0]:w1_index[-1]]
#     f1 = flux[w1_index[0]:w1_index[-1]]

#     f1_norm = []
#     for i, value in enumerate(f1):
#         f = value / (np.median(f1))
#         f1_norm.append(f)

#     return w1_norm, f1_norm



### Normalize 2 chunks of spectra (doublet)
def norm_flux_doublet(wave, flux, fluxerr, chunk1_list, chunk2_list):
    diff0_1, diff1_1 = np.absolute(np.array(wave) - chunk1_list[0]), np.absolute(np.array(wave) - chunk1_list[-1])
    w1_index = [diff0_1.argmin(), diff1_1.argmin()]
    w1_norm = wave[w1_index[0]:w1_index[-1]]
    f1 = flux[w1_index[0]:w1_index[-1]]
    f1err = fluxerr[w1_index[0]:w1_index[-1]]

    f1_norm, f1err_norm = [], []
    for i, value in enumerate(f1):
        value_err = f1err[i]
        f = value / (np.median(f1))
        f_err = (value_err / (np.median(f1)))
        f1_norm.append(f)
        f1err_norm.append(f_err)

    diff0_2, diff1_2 = np.absolute(np.array(wave) - chunk2_list[0]), np.absolute(np.array(wave) - chunk2_list[-1])
    w2_index = [diff0_2.argmin(), diff1_2.argmin()]
    w2_norm = wave[w2_index[0]:w2_index[-1]]
    f2 = flux[w2_index[0]:w2_index[-1]]
    f2err = fluxerr[w2_index[0]:w2_index[-1]]

    f2_norm, f2err_norm = [], []
    for i, value in enumerate(f2):
        value_err = f2err[i]
        f = value / (np.median(f2))
        f_err = (value_err / (np.median(f2)))
        f2_norm.append(f)
        f2err_norm.append(f_err)

    return w1_norm, f1_norm, f1err_norm, w2_norm, f2_norm, f2err_norm