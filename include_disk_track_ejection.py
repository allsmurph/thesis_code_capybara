#%%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import rebound
import astropy.constants as const
from multiprocessing import Pool
from tqdm import tqdm 
from joblib import Parallel, delayed
import os 
import time
import netCDF4
import glob

jtos = const.M_jup / const.M_sun

R_b = 2.72 * const.R_jup.to('au').value
R_c = 2.04 * const.R_jup.to('au').value
R_star = 1.26 * const.R_sun.to('au').value
#%%ä

N_particles = 100

def simulation(tmax, particle_indices, core_id):

    times = np.linspace(0, tmax, int(tmax*5))
    '''
    Run a rebound simulation for specific particle indices for tmax years
    '''
    start = time.time()

    def create_unique_hash(index, coreid):
        return int(index*1000 + coreid)

    sim = rebound.Simulation()
    sim.units = ['msun', 'yr', 'AU']

    sim.add(m = 0.967, hash='star')

    m_b = 7
    m_c = 4.4

    #work in disk inclination
    sim.add(m = m_b * jtos, a = 20.8, e = 0.17, omega = np.radians(161), Omega = np.radians(169.7),
            f = np.random.rand()*2.*np.pi,
            inc = np.radians(131-128.3), hash='pb')
    sim.add(m = m_c * jtos, a = 34.3, e = 0.037, omega = np.radians(53), Omega= np.radians(161.7),
            f = np.random.rand()*2.*np.pi,
            inc = np.radians(130.5-128.3), hash='pc')

    for i in particle_indices:
        unique_hash = create_unique_hash(i, core_id)
        a1 = np.random.uniform(54, 87) 
        inc = np.random.uniform(0, 10)
        sim.add(a=a1, inc=np.radians(inc), f=np.random.rand()*2.*np.pi, e=0, hash=unique_hash)

    sim.N_active = 3
    sim.move_to_com()

    trajectories = {create_unique_hash(i, core_id) : {'ejected': False, 'migrated': False} for i in particle_indices}
    
    filename = f'core_results/tracks_core_{core_id}_{tmax}_yr_{N_particles}_ptcls_short_test.nc'

    with netCDF4.Dataset(filename, 'w') as ncfile:
        ncfile.createDimension('times_to_save', 2)
        ncfile.createDimension('time', len(times))
        ncfile.createDimension('particle', None)
        ncfile.createDimension('ejected_p', None) 
        ncfile.createDimension('massive_p', 3)
        ncfile.createDimension('saved_parameters', 7)
        ncfile.createDimension('saved_ej_param', 8)
        #ncfile.createDimension('string_length', 12)
        ncfile.createDimension('migrated_p', None)

        times_var = ncfile.createVariable('times', 'f4', ('time',))
        test_particles_var = ncfile.createVariable('test_particles', 'f4', ('times_to_save', 'particle', 'saved_parameters'))
        massive_bods_var = ncfile.createVariable('massive_bodies', 'f4', ('times_to_save', 'massive_p', 'saved_parameters'))
        ejected_var = ncfile.createVariable('ejected', 'f4', ('ejected_p', 'saved_ej_param'))
        #p_names_var = ncfile.createVariable('particle_names', 'S1', ('particle', 'string_length'))
        migrated_var = ncfile.createVariable('migrated', 'f4', ('migrated_p', 'saved_ej_param') )

        #particle_names = np.array([f'p{i}_c{core_id}' for i in particle_indices], dtype='S12')
                
        # p_names_var[:] = netCDF4.stringtochar(particle_names)
        # p_names_var._Encoding = 'ascii'  #automatic conversion back
        # p_names_var[:] = particle_names
        times_var[:] = times

        ncfile.description = f'simulation results from core {core_id}'
        ncfile.history = 'created' + time.ctime(time.time())

        count_ejected = 0
        count_migrated = 0

        for t, tid in enumerate(times):
            
            hashlist = []
            for i in range(len(sim.particles)-3):
                hashlist.append(sim.particles[i+3].hash)

            if t > 0:
                dt = times[t] - times[t-1]
                sim.integrate(sim.t + dt)
            
            to_remove = []

            if t == 0:
                for j, hash in enumerate(['star', 'pb', 'pc']):
                    p = sim.particles[hash]
                    if hash != 'star':
                        massive_bods_var[0, j, :] = [p.x, p.y, p.z, p.e, p.a, p.inc, 22222]
                    else:
                        massive_bods_var[0, j, :] = [p.x, p.y, p.z, 0, 0, 0, 11111]

            elif tid == tmax:
                for j, hash in enumerate(['star', 'pb', 'pc']):
                    p = sim.particles[hash]
                    if hash != 'star':
                        massive_bods_var[1, j, :] = [p.x, p.y, p.z, p.e, p.a, p.inc, 22222]
                    else:
                        massive_bods_var[1, j, :] = [p.x, p.y, p.z, 0, 0, 0, 11111]

            for i, hash in enumerate(hashlist):
                p = sim.particles[hash]
                #save where particles start off too
                R = np.sqrt(p.x**2 + p.y**2 + p.z**2)

                if p.e > 1 and R > 200 and not trajectories[hash.value]['ejected']:

                    trajectories[hash.value]['ejected'] = True
                    ejected_var[count_ejected, :] = [tid, p.x, p.y, p.z, p.e, p.a, p.inc, p.hash.value] 
                    to_remove.append(hash.value)
                    count_ejected += 1

                else:
                    if t == 0:
                        test_particles_var[0, i, :] = [p.x, p.y, p.z, p.e, p.a, p.inc, p.hash.value]
                    elif t == tmax:
                        test_particles_var[1, i, :] = [p.x, p.y, p.z, p.e, p.a, p.inc, p.hash.value]


                if np.abs(p.a) < 18 and p.e < 1 and not trajectories[hash.value]['migrated'] and not trajectories[hash.value]['ejected']:
                    trajectories[hash.value]['migrated'] = True
                    migrated_var[count_migrated,:] = [tid, p.x, p.y, p.z, p.e, p.a, p.inc, p.hash.value]
                    count_migrated += 1

            for h in to_remove:
                sim.remove(hash=h)

                              
    end = time.time()

    print(f'core {core_id} done')
    #keep track of which cores finished
    with open(f'progress_tracking_files/progress_{tmax}_yr.txt', 'a') as f:
        f.write(f'Core {core_id} finished and took {(end-start)/3600:.3f} h.\n')


    return filename

#test this function
#%%

def parallelization(N_testparticles, tmax, N_cores):

    with open(f'progress_tracking_files/progress_{tmax}_yr.txt', 'w') as f:
        pass
    
    indices = np.arange(N_testparticles)

    groups = np.array_split(indices, N_cores)
    print('groups:', groups)
    chunk_results = Parallel(n_jobs=N_cores)(
        delayed(simulation)(tmax, group, core_id) for core_id, group in enumerate(groups)
    )
    
    return chunk_results

#%%
if __name__ == '__main__':

    tmax = 5e5
    N_cores = 50
    filenames = parallelization(N_particles, tmax, N_cores)

#%%

#%%

# grazers = []
# for file in filenames:
#     with netCDF4.Dataset(file, 'r') as ncfile:
#         test_particles_a = ncfile['test_particles'][:,:,-1]
#         particle_names = ncfile['particle_names'][:]

#         particle_names = [''.join(name.astype(str)).strip() for name in particle_names]

#         for i, name in enumerate(particle_names):
#             min_a = np.min(test_particles_a[:, i])
#             if np.abs(min_a) < 2:
#                 grazers.append((name, min_a))

# print(grazers)  