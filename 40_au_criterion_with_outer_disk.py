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
import resource

# Limit to 100 GB of virtual memory
limit = 100 * 1024 ** 3
resource.setrlimit(resource.RLIMIT_AS, (limit, limit))

jtos = const.M_jup / const.M_sun

R_b = 2.72 * const.R_jup.to('au').value
R_c = 2.04 * const.R_jup.to('au').value
R_star = 1.26 * const.R_sun.to('au').value

a_b, a_c, a_d = 21.1, 35.3, 10.7
e_b, e_c, e_d = 0.131, 0.033, 0.25
w_b, w_c, w_d = 191.4, 63, 29
i_b, i_c, i_d = 128.7, 128.5, 151
Omega_b, Omega_c, Omega_d = 174.3, 159.8, 144
#m_b, m_c, m_d = 0.7, 2.4, 0.4
m_b, m_c, m_d = 1.5, 5, 0.5

dtor = np.pi / 180
#%%ä

N_particles = 10000

def simulation(tmax, particle_indices, core_id):

    all_times = np.linspace(0, tmax, int(tmax*5))
    midpoint = len(all_times) // 2
    times_no_disk = all_times[:midpoint]
    times_with_disk = all_times[midpoint:]
    start = time.time()


    def create_unique_hash(index, coreid):
        return int(index*1000 + coreid)

    sim = rebound.Simulation()
    sim.units = ['msun', 'yr', 'AU']
    sim.integrator = 'trace'

    sim.add(m = 0.965, hash='star')
    sim.add(m = m_d*jtos, a = a_d, e = e_d, f= 6.274,
            inc = (i_d-128.3)*dtor, omega = w_d*dtor, Omega =  Omega_d*dtor, hash='pd')
    sim.add(m = m_b*jtos, a = a_b, e = e_b, f=2.758,
            omega = w_b*dtor, inc = (i_b-128.3)*dtor, Omega = Omega_b*dtor, hash = 'pb')
    sim.add(m = m_c*jtos, a = a_c, e = e_c, f=5.312,
            omega = w_c*dtor, inc = (i_c-128.3)*dtor, Omega = Omega_c*dtor, hash = 'pc')
    
    sim.move_to_com()

    for t, tid in enumerate(times_no_disk):
            if tid > times_no_disk[0]:
                    dt = times_no_disk[t] - times_no_disk[t-1]
                    sim.integrate(sim.t + dt)


    print(f'Adding test particles now. Orbits: {sim.orbits()}')

    for i in particle_indices:
        unique_hash = create_unique_hash(i, core_id)
        a1 = np.random.uniform(40, 87) 
        inc = np.random.uniform(-10, 10)
        sim.add(a=a1, inc=np.radians(inc), f=np.random.rand()*2.*np.pi, e=0, hash=unique_hash)

    sim.N_active = 4
    sim.move_to_com()

    trajectories = {create_unique_hash(i, core_id) : {'ejected': False, 'migrated': False} for i in particle_indices}
    
    filename = f'core_results/tracks_core_{core_id}_{tmax}_yr_{N_particles}_ptcls_3_planets_outer_disk_40au_check_higher_masses.nc'

    with netCDF4.Dataset(filename, 'w') as ncfile:
        ncfile.createDimension('times_to_save', 2)
        ncfile.createDimension('time', len(times_with_disk))
        ncfile.createDimension('particle', None)
        ncfile.createDimension('ejected_p', None) 
        ncfile.createDimension('massive_p', 4)
        ncfile.createDimension('saved_parameters', 7)
        ncfile.createDimension('saved_ej_param', 8)
        ncfile.createDimension('migrated_p', None)

        times_var = ncfile.createVariable('times', 'f4', ('time',))
        test_particles_var = ncfile.createVariable('test_particles', 'f4', ('times_to_save', 'particle', 'saved_parameters'))
        massive_bods_var = ncfile.createVariable('massive_bodies', 'f4', ('times_to_save', 'massive_p', 'saved_parameters'))
        ejected_var = ncfile.createVariable('ejected', 'f4', ('ejected_p', 'saved_ej_param'))
        migrated_var = ncfile.createVariable('migrated', 'f4', ('migrated_p', 'saved_ej_param') )

        times_var[:] = times_with_disk

        ncfile.description = f'simulation results from core {core_id}. I am using values from Trevascus et al. 2025. Integrator=TRACE. Masses: 0.5, 1.5, 5. I gave specific phases to the planets: 6.274, 2.758, 5.312. Add particles after 2.5Myr to stay consisten with the previous sim.'
        ncfile.history = 'created' + time.ctime(time.time())

        count_ejected = 0
        count_migrated = 0

        for t, tid in enumerate(times_with_disk):
            
            # if tid == 0:
            #     print('just started!')

            # if tid == 1000000:
            #     print('20% there')
            
            # elif tid == 2000000:
            #     print('40% there...')

            # elif tid == 2500000:
            #     print('half way there!')

            # elif tid == 3000000:
            #     print('60% there...')

            # elif tid == 4000000:
            #     print('80% there...')

            hashlist = []
            for i in range(len(sim.particles)-4):
                hashlist.append(sim.particles[i+4].hash)

            if tid > times_with_disk[0]:
                dt = times_with_disk[t] - times_with_disk[t-1]
                sim.integrate(sim.t + dt)
            
            to_remove = []

            if tid == times_with_disk[0]:
                for j, hash in enumerate(['star', 'pb', 'pc', 'pd']):
                    p = sim.particles[hash]
                    if hash != 'star':
                        massive_bods_var[0, j, :] = [p.x, p.y, p.z, p.e, p.a, p.inc, 22222]
                    else:
                        massive_bods_var[0, j, :] = [p.x, p.y, p.z, 0, 0, 0, 11111]

            elif tid == tmax:
                for j, hash in enumerate(['star', 'pb', 'pc', 'pd']):
                    p = sim.particles[hash]
                    if hash != 'star':
                        massive_bods_var[1, j, :] = [p.x, p.y, p.z, p.e, p.a, p.inc, 22222]
                    else:
                        massive_bods_var[1, j, :] = [p.x, p.y, p.z, 0, 0, 0, 11111]

            for i, hash in enumerate(hashlist):
                p = sim.particles[hash]

                R = p.a*(1+p.e)
                dist = np.sqrt(p.x**2 + p.y**2 + p.z**2)
        
                if p.e >= 1 and dist > 200 and not trajectories[hash.value]['ejected']:

                    trajectories[hash.value]['ejected'] = True
                    ejected_var[count_ejected, :] = [tid, p.x, p.y, p.z, p.e, p.a, p.inc, p.hash.value] 
                    to_remove.append(hash.value)
                    count_ejected += 1

                else:
                    if tid == times_with_disk[0]:
                        test_particles_var[0, i, :] = [p.x, p.y, p.z, p.e, p.a, p.inc, p.hash.value]
                    elif t == tmax:
                        test_particles_var[1, i, :] = [p.x, p.y, p.z, p.e, p.a, p.inc, p.hash.value]

                #changed on 22/05 and reran ("corrected.nc" files)

                #corrected_2.nc uses p.a < 18.
                if p.a < 18 and p.e < 1 and not trajectories[hash.value]['migrated']:
                    trajectories[hash.value]['migrated'] = True
                    migrated_var[count_migrated,:] = [tid, p.x, p.y, p.z, p.e, p.a, p.inc, p.hash.value]
                    count_migrated += 1

            for h in to_remove:
                sim.remove(hash=h)

                              
    end = time.time()

    print(f'core {core_id} done')
    with open(f'progress_tracking_files/progress_{tmax}_yr.txt', 'a') as f:
        f.write(f'Core {core_id} finished and took {(end-start)/3600:.3f} h.\n')

    return filename

#test this function
#%%
def prompt():
    confirmation = input('Ally, did you remember to change the name of your file and update file description? Type "YES" if so.')
    if confirmation.upper() == 'YES':
        print('Okay, I believe you, I will now simulate your little planets.')
        return True
    else:
        return False
    
def parallelization(N_testparticles, tmax, N_cores):

    with open(f'progress_tracking_files/progress_{tmax}_yr.txt', 'w') as f:
        pass
    
    indices = np.arange(N_testparticles)

    groups = np.array_split(indices, N_cores)
    chunk_results = Parallel(n_jobs=N_cores)(
        delayed(simulation)(tmax, group, core_id) for core_id, group in enumerate(groups)
    )
    
    return chunk_results

#%%
if __name__ == '__main__':

    tmax = 5e6
    N_cores = 50
    
    if prompt():
        filenames = parallelization(N_particles, tmax, N_cores)
    else:
        print('I think you should change your file name first. You are welcome.')

