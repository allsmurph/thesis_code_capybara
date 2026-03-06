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

dtor = np.pi / 180
#%%ä

def simulation(m_d, m_b, m_c, tmax, particle_indices, core_id, a_group):
    times = np.linspace(0, tmax, int(tmax*5))
    '''
    Run a rebound simulation for specific particle indices for tmax years
    '''
    start = time.time()

    def create_unique_hash(index, coreid):
        return int(index*1000 + coreid)

    sim = rebound.Simulation()
    sim.units = ['msun', 'yr', 'AU']
    sim.collision = 'line'

    sim.add(m = 0.965, x=0, y= 0, z = 0, hash='star', r = R_star)
    sim.add(m = m_d*jtos, a = a_d, e = e_d, f=np.random.rand()*2.*np.pi,
            inc = (i_d-128.3)*dtor, omega = w_d*dtor, Omega =  Omega_d*dtor, hash='pd')
    sim.add(m = m_b*jtos, a = a_b, e = e_b, f=np.random.rand()*2.*np.pi,
            omega = w_b*dtor, inc = (i_b-128.3)*dtor, Omega = Omega_b*dtor, hash = 'pb', r=R_b)
    sim.add(m = m_c*jtos, a = a_c, e = e_c, f=np.random.rand()*2.*np.pi,
            omega = w_c*dtor, inc = (i_c-128.3)*dtor, Omega = Omega_c*dtor, hash = 'pc', r=R_c)
    
    for i, a in zip(particle_indices, a_group):
        unique_hash = create_unique_hash(i, core_id)
        inc = np.random.uniform(-10, 10)
        sim.add(a=a, inc=np.radians(inc), f=np.random.rand()*2.*np.pi, e=0, r=6.68e-9, hash=unique_hash)
    sim.N_active = 4
    sim.move_to_com()

    trajectories = {create_unique_hash(i, core_id) : {'ejected': False, 'migrated': False, 'collided': False} for i in particle_indices}

    filename = f'core_outputs_yr2/core_{core_id}_{tmax}_yr_{N_particles}_test.nc'
    #filename = f'core_outputs_yr2/test_{core_id}.nc'
    with netCDF4.Dataset(filename, 'w') as ncfile:
        ncfile.createDimension('times_to_save', 2)
        ncfile.createDimension('time', len(times))
        ncfile.createDimension('particle', None)
        ncfile.createDimension('ejected_p', None) 
        ncfile.createDimension('massive_p', 4)
        ncfile.createDimension('saved_parameters', 8)
        ncfile.createDimension('saved_ej_param', 9)
        ncfile.createDimension('migrated_p', None)
        ncfile.createDimension('collided_p', None)

        times_var = ncfile.createVariable('times', 'f4', ('time',))
        test_particles_var = ncfile.createVariable('test_particles', 'f4', ('times_to_save', 'particle', 'saved_parameters'))
        massive_bods_var = ncfile.createVariable('massive_bodies', 'f4', ('times_to_save', 'massive_p', 'saved_parameters'))
        ejected_var = ncfile.createVariable('ejected', 'f4', ('ejected_p', 'saved_ej_param'))
        migrated_var = ncfile.createVariable('migrated', 'f4', ('migrated_p', 'saved_ej_param'))
        collided_var = ncfile.createVariable('collided', 'f4', ('collided_p', 'saved_ej_param'))

        times_var[:] = times

        ncfile.description = f'simulation results from core {core_id}. Trevascus 2025 values but my masses: 0.5, 1.5, 3. "mig" when a<1 au.'
        ncfile.history = 'created' + time.ctime(time.time())

        count_ejected = 0
        count_migrated = 0
        count_collided = 0
        to_remove = []

        def collision_resolve(sim_pointer, collision):
            nonlocal count_collided, to_remove

            sim = sim_pointer.contents
            p1 = sim.particles[collision.p1]
            p2 = sim.particles[collision.p2]

            h1 = p1.hash
            h2 = p2.hash

            if (sim.particles[h1].m != 0 and sim.particles[h2].m == 0) or \
                (sim.particles[h1].m == 0 and sim.particles[h2].m != 0):
                
                hash_val = h1 if sim.particles[h2].m !=0 else h2

                if not trajectories[hash_val.value]['collided']:
                    trajectories[hash_val.value]['collided'] = True
                    print(f'collided! {p1.hash, p2.hash}')
                    # Save parameters in NetCDF
                    pp = sim.particles[hash_val]
                    collided_var[count_collided, :] = [sim.t, pp.x, pp.y, pp.z, pp.e, pp.a, pp.inc, pp.f, pp.hash.value]


            return 0
        

        sim.collision = 'line'
        sim.collision_resolve = collision_resolve

        for t, tid in enumerate(times):

            to_remove.clear()            
            hashlist = []
            for i in range(len(sim.particles)-4):
                hashlist.append(sim.particles[i+4].hash)

            if t > 0:
                dt = times[t] - times[t-1]
                sim.integrate(sim.t + dt)
            
            apo_dist, peri_dist, eccs = [], [], []
            for i in ['pb', 'pc', 'pd']:
                apo_dist.append(sim.particles[i].a * (1+sim.particles[i].e))
                peri_dist.append(sim.particles[i].a * (1-sim.particles[i].e))
                eccs.append(sim.particles[i].e)

            apo_dist = np.array(apo_dist)
            peri_dist = np.array(peri_dist)
            eccs = np.array(eccs)

            c1 = np.all(eccs < 1)
            c2 = (apo_dist[2] < peri_dist[0]) and (apo_dist[0] < peri_dist[1] ) 

            if not (c1 and c2):
                with open(f'progress_tracking_files/failed_cores.txt', 'a') as f:
                    f.write(f'File {filename} failed at and t = {tid} yr.\n')

                print(f'Core {core_id} failed.')
                return None

            if t == 0:
                for j, hash in enumerate(['star', 'pb', 'pc', 'pd']):
                    p = sim.particles[hash]
                    if hash != 'star':
                        massive_bods_var[0, j, :] = [p.x, p.y, p.z, p.e, p.a, p.inc, p.f, 22222]
                    else:
                        massive_bods_var[0, j, :] = [p.x, p.y, p.z, 0, 0, 0, 0, 11111]

            elif tid == tmax:
                for j, hash in enumerate(['star', 'pb', 'pc', 'pd']):
                    p = sim.particles[hash]
                    if hash != 'star':
                        massive_bods_var[1, j, :] = [p.x, p.y, p.z, p.e, p.a, p.inc, p.f, 22222]
                    else:
                        massive_bods_var[1, j, :] = [p.x, p.y, p.z, 0, 0, 0, 0, 11111]

            for i, hash in enumerate(hashlist):

                p = sim.particles[hash]
                traj = trajectories[hash.value]

                R = p.a*(1+p.e)
                dist = np.sqrt(p.x**2 + p.y**2 + p.z**2)
        
                if p.e >= 1 and dist > 200 and not traj['ejected']:
                    print('ejected')
                    trajectories[hash.value]['ejected'] = True
                    ejected_var[count_ejected, :] = [tid, p.x, p.y, p.z, p.e, p.a, p.inc, p.f, p.hash.value] 
                    to_remove.append(hash.value)
                    count_ejected += 1
                #changed on 22/05 and reran ("corrected.nc" files)

                #corrected_2.nc uses p.a < 18.

                if p.a < 1 and p.e < 1 and not traj['migrated']:
                    print('migrated')
                    trajectories[hash.value]['migrated'] = True
                    migrated_var[count_migrated,:] = [tid, p.x, p.y, p.z, p.e, p.a, p.inc, p.f, p.hash.value]
                    count_migrated += 1
                    to_remove.append(hash.value)

                if traj['collided']:
                    print('collided')
                    count_collided += 1
                    to_remove.append(hash.value)

                if not traj['migrated'] and not traj['ejected'] and not traj['collided']:
                    if t == times[0]:
                        test_particles_var[0, i, :] = [p.x, p.y, p.z, p.e, p.a, p.inc, p.f, p.hash.value]

            for h in to_remove:
                sim.remove(hash=h)
                print(f'following particles removed: {h}')

            #save all other particles left at final timestep:
            

        print(f'At {sim.t} yr:', len(sim.particles)-4, 'particles')
        for i in range(len(sim.particles)-4):
            test_particles_var[1, i, :] = [p.x, p.y, p.z, p.e, p.a, p.inc, p.f, p.hash.value]

    
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
    a_vals = np.random.uniform(21.1, 21.2, N_testparticles)

    indices = np.arange(N_testparticles)
    sorted_idx = np.argsort(a_vals)   # indices of particles sorted by semimajor axis
    a_vals = a_vals[sorted_idx]       # sorted semimajor axes
    indices = indices[sorted_idx]   

    groups = np.array_split(indices, N_cores)
    a_groups = np.array_split(a_vals, N_cores)
    chunk_results = Parallel(n_jobs=N_cores)(
        delayed(simulation)(0.5, 1.5, 3, tmax, group, core_id, a_group) for core_id, (group, a_group) in enumerate(zip(groups, a_groups))
    )
    
    return chunk_results

#%%
if __name__ == '__main__':

    tmax = 1e5
    N_cores = 80
    N_particles = 800
    if prompt():
        filenames = parallelization(N_particles, tmax, N_cores)
    else:
        print('I think you should change your file name first. You are welcome.')

