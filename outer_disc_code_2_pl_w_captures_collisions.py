#%%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import rebound
import astropy.constants as const
from multiprocessing import Pool
import requests
from tqdm import tqdm 
from joblib import Parallel, delayed
import os 
import time
import netCDF4
import glob
import resource

# Limit to 100 GB of virtual memory,noone likes me and i know why

limit = 100 * 1024 ** 3
resource.setrlimit(resource.RLIMIT_AS, (limit, limit))

jtos = const.M_jup / const.M_sun

R_b = 2.72 * const.R_jup.to('au').value
R_c = 2.04 * const.R_jup.to('au').value
R_star = 1.26 * const.R_sun.to('au').value

# a_b, a_c, a_d = 21.1, 35.3, 10.7
# e_b, e_c, e_d = 0.131, 0.033, 0.25
# w_b, w_c, w_d = 191.4, 63, 29
# i_b, i_c, i_d = 128.7, 128.5, 151
# Omega_b, Omega_c, Omega_d = 174.3, 159.8, 144
a_b, a_c = 20.7, 33.9
e_b, e_c = 0.16, 0.042
i_b, i_c = 130.6, 129.8
w_b, w_c = 190, 77
Omega_b, Omega_c = 176, 158
#m_b, m_c, m_d = 0.7, 2.4, 0.4

dtor = np.pi / 180
#%%ä


def simulation(m_b, m_c, tmax, particle_indices, core_id, a_group):

    times = np.linspace(0, tmax, int(tmax*5))

    '''
    Run a rebound simulation for specific particle indices for tmax years
    '''
    start = time.time()

    def create_unique_hash(index, coreid):
        return int(index*1000 + coreid)

    sim = rebound.Simulation()
    sim.units = ['msun', 'yr', 'AU']

    sim.add(m = 0.952, x=0, y= 0, z = 0, hash='star')
    #sim.add(m = m_d*jtos, a = a_d, e = e_d, f=np.random.rand()*2.*np.pi,
    #        inc = (i_d-128.3)*dtor, omega = w_d*dtor, Omega =  Omega_d*dtor, hash='pd')
    sim.add(m = m_b*jtos, a = a_b, e = e_b, f=np.random.rand()*2.*np.pi, primary=sim.particles['star'],
            omega = w_b*dtor, inc = (i_b-128.3)*dtor, Omega = Omega_b*dtor, hash = 'pb')
    sim.add(m = m_c*jtos, a = a_c, e = e_c, f=np.random.rand()*2.*np.pi, primary=sim.particles['star'],
            omega = w_c*dtor, inc = (i_c-128.3)*dtor, Omega = Omega_c*dtor, hash = 'pc')
    
    for i, a in zip(particle_indices, a_group):
        unique_hash = create_unique_hash(i, core_id)
        inc = np.random.uniform(-10, 10)
        sim.add(a=a, inc=np.radians(inc), f=np.random.rand()*2.*np.pi, e=0, 
                primary=sim.particles['star'], hash=unique_hash)
        

    sim.N_active = 3
    sim.move_to_com()

    trajectories = {create_unique_hash(i, core_id) : {'ejected': False, 'migrated': False,
                                                      'collided': False, 'captured': False,
                                                    'captured_counter':0, 'captured_t0':None,
                                                      'migrated_peri': False} for i in particle_indices}
    
    #filename = f'core_outputs_yr2/core_{core_id}_{tmax}_yr_{N_particles}_2_pl_w_captures_and_saving_every_100_yrs.nc'
    filename = f'core_outputs_yr2/tests/core_{core_id}_test.nc'
    with netCDF4.Dataset(filename, 'w') as ncfile:

        sampling_period = 5000 #every x/5 = 1000
        s_times = [time for i, time in enumerate(times) if i%sampling_period==0]
        n_saved_times = len(s_times)
        ncfile.createDimension('times_to_save', n_saved_times)
        #ncfile.createDimension('time', len(times))
        ncfile.createDimension('particle', sim.N-1)
        ncfile.createDimension('ejected_p', None) 
        ncfile.createDimension('massive_p', 3)
        ncfile.createDimension('saved_parameters', 8)
        ncfile.createDimension('saved_ej_param', 11)
        ncfile.createDimension('saved_collided_p', 12) #track which planet it collides with
        #ncfile.createDimension('migrated_p', None)
        ncfile.createDimension('migrated_peri_p', None)
        ncfile.createDimension('all_saved_times', None)
        ncfile.createDimension('collided_p', None)
        ncfile.createDimension('captured_p', None)

        #times_var = ncfile.createVariable('times', 'f4', ('time',))
        test_particles_var = ncfile.createVariable('test_particles', 'f4', ('times_to_save', 'particle', 'saved_parameters'))
        massive_bods_var = ncfile.createVariable('massive_bodies', 'f4', ('all_saved_times', 'massive_p', 'saved_ej_param'))
        ejected_var = ncfile.createVariable('ejected', 'f4', ('ejected_p', 'saved_ej_param'))
        #migrated_var = ncfile.createVariable('migrated', 'f4', ('migrated_p', 'saved_ej_param') )
        migrated_peri_var = ncfile.createVariable('migrated_peri', 'f4', ('migrated_peri_p', 'saved_ej_param') )
        collided_var = ncfile.createVariable('collided', 'f4', ('collided_p', 'saved_collided_p'))
        captured_var = ncfile.createVariable('captured', 'f4', ('captured_p', 'saved_collided_p'))
        
        #times_var[:] = times

        ncfile.description = f'simulation results from core {core_id}. Trevascus 2025 values (inc masses).2 planets!! "peri mig" when a(1-e)<18. do not delete particles when migrated. tracking captures and collisions. saving every 100 yrs'
        ncfile.history = 'created' + time.ctime(time.time())

        count_ejected = 0
        count_migrated_peri = 0
        count_collided = 0
        count_captured = 0
        save_t_index = 0

        def collision_resolve(sim_pointer, collision):
            nonlocal count_collided

            sim = sim_pointer.contents
            p1 = sim.particles[collision.p1]
            p2 = sim.particles[collision.p2]

            h1 = p1.hash
            h2 = p2.hash

            if (sim.particles[h1].m != 0 and sim.particles[h2].m == 0) or \
                (sim.particles[h1].m == 0 and sim.particles[h2].m != 0):
                
                hash_val = h1 if sim.particles[h2].m !=0 else h2
                planet_hash = h1 if sim.particles[h2].m == 0 else h2

                if not trajectories[hash_val.value]['collided']:
                    trajectories[hash_val.value]['collided'] = True
                    count_collided += 1

                    print(f'collided! {p1.hash, p2.hash}')
                    # Save parameters in NetCDF
                    pp = sim.particles[hash_val]
                    collided_var[count_collided, :] = [sim.t, pp.x, pp.y, pp.z, pp.e, pp.a, pp.inc, pp.f, pp.Omega, pp.omega, pp.hash.value, planet_hash]

            return 0
        

        sim.collision = 'line'
        sim.collision_resolve = collision_resolve

        def hill_radius(p):
            return p.a * (1 - p.e) * (p.m / (3 * sim.particles['star'].m))**(1/3)
        
        hill_pb = hill_radius(sim.particles['pb'])
        hill_pc = hill_radius(sim.particles['pc'])


        for t, tid in enumerate(times):
            
            if t > 0:
                dt = times[t] - times[t-1]
                sim.integrate(sim.t + dt)
                hill_pb = hill_radius(sim.particles['pb']) #update 
                hill_pc = hill_radius(sim.particles['pc']) #update 
            
            apo_dist = np.array([sim.particles['pb'].a * (1+sim.particles['pb'].e),
                                sim.particles['pc'].a * (1+sim.particles['pc'].e)])
            peri_dist = np.array([sim.particles['pb'].a * (1-sim.particles['pb'].e),
                                sim.particles['pc'].a * (1-sim.particles['pc'].e)])
            eccs = np.array([sim.particles['pb'].e, sim.particles['pc'].e])


            c1 = np.all(eccs < 1)
            c2 = apo_dist[0] < peri_dist[1] 

            if not (c1 and c2):
                with open(f'progress_tracking_files/failed_cores_2pl_outerdisc.txt', 'a') as f:
                    f.write(f'File {filename} failed at and t = {tid} yr.\n')

                print(f'Core {core_id} failed.')
                return None
            
            
            to_remove = []

            # if tid == 0:
            #     for j, hash in enumerate(['star', 'pb', 'pc']):
            #         p = sim.particles[hash]
            #         if hash != 'star':
            #             massive_bods_var[0, j, :] = [t, p.x, p.y, p.z, p.e, p.a, p.inc, p.f, p.Omega, p.omega, 22222]
            #         else:
            #             massive_bods_var[0, j, :] = [t, p.x, p.y, p.z,0, 0, 0, 0, 0, 0, 11111]


            # if t == len(times) - 1:
            #     for j, hash in enumerate(['star', 'pb', 'pc']):
            #         p = sim.particles[hash]
            #         if hash != 'star':
            #             massive_bods_var[count_migrated_peri+2, j, :] = [tid, p.x, p.y, p.z, p.e, p.a, p.inc, p.f, p.Omega, p.omega, 22222]
            #         else:
            #             massive_bods_var[count_migrated_peri+2, j, :] = [tid, p.x, p.y, p.z,0, 0, 0, 0, 0, 0, 11111]


            for p in sim.particles[sim.N_active:]:
                h = p.hash.value
                traj = trajectories[h]

                dist = np.sqrt(p.x**2 + p.y**2 + p.z**2)
        
                if p.e >= 1 and dist > 200 and not traj['ejected']:
                    traj['ejected'] = True
                    ejected_var[count_ejected, :] = [tid, p.x, p.y, p.z, p.e, p.a, p.inc, p.f, p.Omega, p.omega, h] 
                    to_remove.append(h)
                    count_ejected += 1


                # if p.e < 1 and dist < 200: 
                #     if t == 0:
                #         test_particles_var[0, i, :] = [p.x, p.y, p.z, p.e, p.a, p.inc, p.f, p.hash.value]

                #need to check they are also within the orbits of outer planets!!
                if (p.a*(1-p.e)) <= 18 and p.e < 1 and not traj['migrated_peri']:
                    traj['migrated_peri'] = True
                    migrated_peri_var[count_migrated_peri,:] = [tid, p.x, p.y, p.z, p.e, p.a, p.inc, p.f, p.Omega, p.omega, p.hash.value]
                    for j, plhash in enumerate(['star', 'pb', 'pc']):
                        pl = sim.particles[plhash]
                        if plhash != 'star':
                            massive_bods_var[count_migrated_peri, j, :] = [tid, pl.x, pl.y, pl.z, pl.e, pl.a, pl.inc, pl.f, pl.Omega, pl.omega, 22222]
                        else:
                            massive_bods_var[count_migrated_peri, j, :] = [tid, pl.x, pl.y, pl.z, 0, 0, 0, 0, 0, 0, 11111]

                    #to_remove.append(hash.value)
                    count_migrated_peri += 1

                
                d_pb = np.sqrt((p.x - sim.particles['pb'].x)**2 +
                (p.y - sim.particles['pb'].y)**2 +
                (p.z - sim.particles['pb'].z)**2)

                d_pc = np.sqrt((p.x - sim.particles['pc'].x)**2 +
                                (p.y - sim.particles['pc'].y)**2 +
                                (p.z - sim.particles['pc'].z)**2)

                inside_pb = d_pb < hill_pb
                inside_pc = d_pc < hill_pc

                capturing_planet = None

                if inside_pb:
                    capturing_planet = sim.particles['pb']
                elif inside_pc:
                    capturing_planet = sim.particles['pc']

                if capturing_planet is not None:
                    if traj['captured_t0'] is None:
                        traj['captured_t0'] = sim.t
                    traj['captured_counter'] +=1

                else: #reset if not actually captured
                    traj['captured_counter'] = 0
                    traj['captured_t0'] = None

                if not traj['captured'] and not traj['ejected']:
                    if (traj['captured_counter'] > 50000): # 10 thousand years
                        print(f'captured by {capturing_planet.hash.value}', p.a, 'r:', dist)
                        traj['captured'] = True
                        captured_var[count_captured, :] = [tid, p.x, p.y, p.z, p.e, p.a, p.inc, p.f, p.Omega, p.omega, h, capturing_planet.hash.value]
                        to_remove.append(h)
                        count_captured += 1

                
                if traj['collided']:
                    print('collided')
                    to_remove.append(h)


            for h in to_remove:
                sim.remove(hash=h)

                    
            if t % sampling_period == 0:
                for i, p in enumerate(sim.particles[1:]):
                    test_particles_var[save_t_index, i, :] = [tid, p.x, p.y, p.z, p.e, p.a, p.inc, p.hash.value]
                save_t_index += 1

                              
    end = time.time()

    print(f'core {core_id} done')
    with open(f'progress_tracking_files/progress_{tmax}_yr_2pl_outerdisc.txt', 'a') as f:
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

    with open(f'progress_tracking_files/progress_{tmax}_yr_2pl_outerdisc.txt', 'w') as f:
        pass
    a_vals = np.random.uniform(54, 87, N_testparticles)

    indices = np.arange(N_testparticles)
    sorted_idx = np.argsort(a_vals)   # indices of particles sorted by semimajor axis
    a_vals = a_vals[sorted_idx]       # sorted semimajor axes
    indices = indices[sorted_idx]   

    groups = np.array_split(indices, N_cores)
    a_groups = np.array_split(a_vals, N_cores)
    print(a_groups)
    chunk_results = Parallel(n_jobs=N_cores)(
        delayed(simulation)(1.4, 6.4, tmax, group, core_id, a_group) for core_id, (group, a_group) in enumerate(zip(groups, a_groups))
    )
    
    return chunk_results

#%%
from notify_run import Notify

notify = Notify()
print(notify.endpoint)
if __name__ == '__main__':

    N_particles = 100

    tmax = 1000
    N_cores = 5

    notify.send('Script started')

    if prompt():
        filenames = parallelization(N_particles, tmax, N_cores)
    else:
        print('I think you should change your file name first. You are welcome.')

    notify.send('Script done! ✅')

