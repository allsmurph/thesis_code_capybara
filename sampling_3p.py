#%%
import numpy as np
import rebound
import matplotlib.pyplot as plt
import itertools
from astropy import constants as const # type: ignore
from tqdm import tqdm # type: ignore
import glob
from joblib import Parallel, delayed # type: ignore
import netCDF4 # type: ignore
import time 
import sys
import getpass
import os 

# a_b, a_c = 20.8, 34.3
# e_b, e_c = 0.17, 0.037
# w_b, w_c = 161, 53
# i_b, i_c = 131, 130.5
# Omega_b, Omega_c = 169.7, 161.7

#MMR values within 95% CI:
#ab, ac, ad = 20.67, 33.11, 12.9

#MMR values exact:
#ab, ac, ad = 20.4775, 32.35, 12.9

#trevascus et al. 2025: 
#ab, ac, ad = 21.1, 35.3, 10.7

a_b, a_c, a_d = 21.1, 35.3, 10.7
e_b, e_c, e_d = 0.131, 0.033, 0.25
i_b, i_c, i_d = 128.7, 128.5, 151
w_b, w_c, w_d = 191.4, 63, 29
Omega_b, Omega_c, Omega_d = 174.3, 159.8, 144

jtos = const.M_jup / const.M_sun

#%%

###################   CHANGE FILE NAME!!!   #######################


def mass_sampling(tmax, mass_combos, core_id, fname):
    times = np.linspace(0, tmax, int(tmax*5))

    filename = fname + f'core_{core_id}_3p_with_ecc_damping.nc'
    
    start = time.time()

    with netCDF4.Dataset(filename, 'w') as file:

        file.createDimension('planets', 3)
        #because we are taking steps of 5, if we only want every 5 years, 
        #gotta be divisible by 50.
        s_times = [time for i, time in enumerate(times) if i%50==0]
        n_saved_times = len(s_times)
        file.createDimension('time', n_saved_times)
        file.createDimension('saved_planet_params', 2)
        #file.createDimension('star_params', 3)
        file.createDimension('mass_combos', len(mass_combos))

        times_var = file.createVariable('times', 'f4', ('time',))
        planets_var = file.createVariable('planets', 'f4', ('time', 'mass_combos', 'planets', 'saved_planet_params'))
        #star_var = file.createVariable('star', 'f4', ('time', 'mass_combos', 'star_params'))

        times_var[:] = s_times

        for m, mass_combo in enumerate(mass_combos):
            mass1, mass2, mass3 = mass_combo[0], mass_combo[1], mass_combo[2]

            sim = rebound.Simulation()
            sim.units = ['msun', 'yr', 'AU']

            sim.add(m = 0.965, hash='star')
            
            sim.add(m = mass3*jtos, a = a_d, e = e_d, omega=np.radians(w_d),
                    f=np.random.rand()*2.*np.pi,
                    inc=np.radians(i_d-128.3), Omega = np.radians(Omega_d), hash='pd')
            
            sim.add(m=mass1*jtos, a = a_b, e = e_b, omega=np.radians(w_b),
                    #look into l parameter - also, isn't phase known?
                    f=np.random.rand()*2.*np.pi,
                    inc=np.radians(i_b-128.3), Omega = np.radians(Omega_b), hash='pb')
            
            sim.add(m=mass2*jtos, a=a_c, e=e_c, omega=np.radians(w_c),
                    f=np.random.rand()*2.*np.pi,
                    inc=np.radians(i_c-128.3), Omega=np.radians(Omega_c), hash='pc')   

            sim.move_to_com()
            
            save_t_index = 0
            for tt, _ in enumerate(times):

                if tt > 0:
                    dt = times[tt] - times[tt-1]
                    sim.integrate(sim.t + dt)

                if tt % 50 == 0:        
                    for j, hash in enumerate(['pb', 'pc', 'pd']):
                        p = sim.particles[hash]
                        #planets_var[save_t_index, m, j, :] = [p.x, p.y, p.z, p.a, p.e, p.inc]
                        planets_var[save_t_index, m, j, :] = [p.e, p.a]

                    # star = sim.particles['star']
                    # star_var[save_t_index, m, :] = [star.x, star.y, star.z]
                    save_t_index += 1

    end = time.time()
    print(f'core {core_id} finished and took {(end-start)/3600:.3f} h. Saved nc file as {filename}')

    # with open(prog_file_name, 'a') as f:
    #     f.write(f'Core {core_id} finished and took {(end-start)/3600:.3f} h.\n')
    # f.close()
    return filename
#%%

def parallelization(tmax, combos, N_cores, filename):

    # with open(prog_file_name, 'w') as f:
    #     pass

    groups = np.array_split(combos, N_cores)
    print('groups:', len(groups), groups)
    chunk_results = Parallel(n_jobs=N_cores)(
        delayed(mass_sampling)(tmax, chunk, core_id, filename) for core_id, chunk in enumerate(groups)
    )
    return chunk_results

def prompt():
    confirmation = input('Ally, did you remember to change the name of your file and description? Type "YES" if so.')
    if confirmation.upper() == 'YES':
        print('Okay, I believe you, I will now simulate your little planets.')
        return True
    else:
        return False
#%%
if __name__ == '__main__':
    
    tmax = 2e6
    masses_b = np.linspace(1, 3, 5)
    masses_c = np.linspace(1, 7, 4)
    masses_d = np.linspace(0.5, 1, 2)
    combinations = list(itertools.product(masses_b, masses_c, masses_d))
    
    N_cores = 40
    if prompt():
        for i in (np.arange(50, 100, 1)):
            #print('doing a fake simulation hehe')
            parallelization(tmax, combinations, N_cores, f'mass_sampling_results/run_{i}_{tmax}_yr_')
    else:
        print('I think you should change your file name first. You are welcome.')
#%%
masses_b = np.linspace(1, 3, 5)
masses_c = np.linspace(1, 7, 4)
masses_d = np.linspace(0.5, 1, 2)

combinations = list(itertools.product(masses_b, masses_c, masses_d))
print(len(combinations))
print(masses_b, masses_c, masses_d)
# %%
# import os

# for i in range(81):  # 0 to 80
#     old_name = f"thesis_code/mass_sampling_results/run_6_2000000.0_yr_core_{i}_3p_v2.nc"
#     new_name = f"thesis_code/mass_sampling_results/run_6_2000000.0_yr_core_{i}_3p_MMR.nc"
#     if os.path.exists(old_name):
#         print('changing name')
#         os.rename(old_name, new_name)