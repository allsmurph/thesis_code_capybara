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

a_b, a_c = 20.7, 33.9
e_b, e_c = 0.16, 0.042
i_b, i_c = 130.6, 129.8
w_b, w_c = 190, 77
Omega_b, Omega_c = 176, 158

jtos = const.M_jup / const.M_sun

#%%

###################   CHANGE FILE NAME!!!   #######################


def mass_sampling(tmax, mass_combos, core_id, fname):
    times = np.linspace(0, tmax, int(tmax*5))

    filename = fname + f'core_{core_id}_2p.nc'
    # if len(sys.argv) > 1:
    #     filename = sys.argv[1]
    #     prog_file_name = sys.argv[2]

    # else:
    #     print('You didnt give your file a name! Thats okay...')
    #     filename = f'new_core_results/mass_sampling_{core_id}_{tmax:.1e}_yr_x.nc'

    start = time.time()

    with netCDF4.Dataset(filename, 'w') as file:

        file.createDimension('planets', 2)
        #because we are taking steps of 5, if we only want every 10 years, 
        #gotta be divisible by 50.
        s_times = [time for i, time in enumerate(times) if i%50==0]
        n_saved_times = len(s_times)
        file.createDimension('time', n_saved_times)
        file.createDimension('saved_planet_params', 5)
        file.createDimension('mass_combos', len(mass_combos))

        planets_var = file.createVariable('planets', 'f4', ('time', 'mass_combos', 'planets', 'saved_planet_params'))

        file.description = f'Trevascus 2025 values - 2 planets!! Mass sampling of b and c redone 30/11/2025. each core deals with a diff combo. Masses for both planets are linspace(1, 11, 6)'

        for m, mass_combo in enumerate(mass_combos):

            
            sim = rebound.Simulation()
            sim.units = ['msun', 'yr', 'AU']

            sim.add(m = 0.952, hash='star')
   
            sim.add(m=mass_combo[0]*jtos, a = a_b, e = e_b, omega=np.radians(w_b),
                    #look into l parameter - also, isn't phase known?
                    f=np.random.rand()*2.*np.pi,
                    inc=np.radians(i_b-128.3), Omega = np.radians(Omega_b), hash='pb')
                   
            sim.add(m = mass_combo[1]*jtos, a = a_c, e = e_c, omega=np.radians(w_c),
                    f=np.random.rand()*2.*np.pi,
                    inc=np.radians(i_c-128.3), Omega = np.radians(Omega_c), hash='pc')
              
            sim.N_active = 3
            sim.move_to_com()
            
            save_t_index = 0
            for tt, tid in enumerate(times):

                if tt > 0:
                    dt = times[tt] - times[tt-1]
                    sim.integrate(sim.t + dt)

                if tt % 50 == 0:        
                    for j, hash in enumerate(['pb', 'pc']):
                        p = sim.particles[hash]
                        #planets_var[save_t_index, m, j, :] = [p.x, p.y, p.z, p.a, p.e, p.inc]
                        planets_var[save_t_index, m, j, :] = [tid, p.m, p.e, p.a, p.inc]

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
    #print('groups:', len(groups), groups)
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
    masses_b = np.linspace(1, 11, 6)
    masses_c = np.linspace(1, 11, 6)
 

    combinations = list(itertools.product(masses_b, masses_c))

    N_cores = 36

    if prompt():
        for i in (np.arange(0, 100, 1)):
            #print('doing a fake simulation hehe')
            parallelization(tmax, combinations, N_cores, f'mass_sampling_results/run_{i}_{tmax}_yr_')
    else:
        print('I think you should change your file name first. You are welcome.')
  
#%%