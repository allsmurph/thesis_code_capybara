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
import reboundx

limit = 100 * 1024 ** 3
resource.setrlimit(resource.RLIMIT_AS, (limit, limit))

def simulation(tmax, a_value):
    times = np.linspace(0, tmax, int(tmax*5))
    '''
    Run a rebound simulation for specific particle indices for tmax years
    '''
    start = time.time()

    def create_unique_hash(index, aval):
        return int(index*1000 + aval)

    sim = rebound.Simulation()
    sim.units = ['msun', 'yr', 'AU']

    sim.add(m = 0.965, x=0, y= 0, z = 0, hash='star')

    ecc_list = np.linspace(0.6, 0.9, 4)
    for i,ecc in enumerate(ecc_list):
        unique_hash = create_unique_hash(i, a_value)
        inc = np.random.uniform(-10,10)
        sim.add(a=a_value, inc=np.radians(inc), f=0, e=ecc, r=6.68e-9, hash=unique_hash)
        #sim.add(a=a_value, inc=np.radians(inc), f=np.pi/2, e=ecc, r=6.68e-9, hash=unique_hash)

    
    sim.N_active = 1
    sim.move_to_com()

    rebx = reboundx.Extras(sim)

    myforce = rebx.load_force("test_force")     
    rebx.add_force(myforce)

    filename = f'core_outputs_yr2/decomposing_GD/a_{a_value}_AU__{len(ecc_list)}_ptcls_5myr.nc'
    with netCDF4.Dataset(filename, 'w') as ncfile:

        #saving every timestep takes too much space, so only save every nth timestep
        sampling_period = 100
        s_times = [time for i, time in enumerate(times) if i%sampling_period==0]
        n_saved_times = len(s_times)
        ncfile.createDimension('times_to_save', n_saved_times)
        ncfile.createDimension('time', len(times))
        ncfile.createDimension('particle', len(ecc_list))
        ncfile.createDimension('saved_parameters', 12)

        times_var = ncfile.createVariable('times', 'f4', ('time',))
        test_particles_var = ncfile.createVariable('test_particles', 'f4', ('times_to_save', 'particle', 'saved_parameters'))
        
        times_var[:] = times

        ncfile.description = f'simulation results from core with a={a_value}. No planets. 10 particles with 10 values of e.'
        ncfile.history = 'created' + time.ctime(time.time())

        save_t_index = 0

        for t, tid in enumerate(times):
            if t > 0:
                dt = times[t] - times[t-1]
                sim.integrate(sim.t + dt)


            if t % sampling_period == 0:

                for i, p in enumerate(sim.particles[sim.N_active:]):
                    test_particles_var[save_t_index, i, :] = [tid, p.x, p.y, p.z, p.vx, p.vy, p.vz, p.e, p.a, p.inc, p.f, p.hash.value]

                save_t_index += 1

            for pt in sim.particles[sim.N_active:]:
                if pt.a < 0:
                    break
 
        print(f'At {sim.t} yr:', len(sim.particles)-sim.N_active, 'particles')
        
    end = time.time()

    print(f'core a={a_value} AU done')
    with open(f'progress_tracking_files/progress_{tmax}_yr.txt', 'a') as f:
        f.write(f'Core with a={a_value} AU finished and took {(end-start)/60:.3f} min.\n')

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
    
def parallelization(tmax, N_cores, a_vals):

    with open(f'progress_tracking_files/progress_{tmax}_yr.txt', 'w') as f:
        pass
    
    chunk_results = Parallel(n_jobs=N_cores)(
        delayed(simulation)( tmax, a_vals[i]) for i in range(N_cores)
    )
    
    return chunk_results

#%%
if __name__ == '__main__':

    tmax = 5e6
    a_vals = np.arange(10,19,1)
    N_cores = len(a_vals)

    if prompt():
        filenames = parallelization(tmax, N_cores, a_vals)
    else:
        print('I think you should change your file name first. You are welcome.')

