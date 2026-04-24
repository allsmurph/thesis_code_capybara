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

radii = 10 * const.R_jup.to('au').value
R_star = 1.26 * const.R_sun.to('au').value
# a_b, a_c, a_d = 21.1, 35.3, 10.7
# e_b, e_c, e_d = 0.131, 0.033, 0.25
# w_b, w_c, w_d = 191.4, 63, 29
# i_b, i_c, i_d = 128.7, 128.5, 151
# Omega_b, Omega_c, Omega_d = 174.3, 159.8, 144

two_planet_case = {
    'star' : {'mass':0.952},
    'pb' : {'mass': 1.4*jtos, 'a': 20.7, 'e': 0.16,
            'inc': np.radians(128.3 - 130.6),
            'omega': np.radians(190), 'Omega': np.radians(176)},
    'pc' : {'mass': 6.4*jtos, 'a': 33.9,
            'e': 0.042, 'inc': np.radians(128.3 - 129.8),
            'omega': np.radians(77), 'Omega': np.radians(158)}
}

three_planet_case = {
    'star' : {'mass':0.965},
    'pb' : {'mass': 0.7*jtos, 'a': 21.1, 'e': 0.131,
            'inc': np.radians(128.3 - 128.7),
            'omega': np.radians(191.4), 'Omega': np.radians(174.3)},
    'pc' : {'mass': 2.4*jtos, 'a': 35.3,
            'e': 0.033, 'inc': np.radians(128.3 - 128.5),
            'omega': np.radians(63), 'Omega': np.radians(159.8)},
    'pd' : {'mass': 0.4*jtos, 'a': 10.7,
            'e': 0.25, 'inc': np.radians(128.3 - 151),
            'omega': np.radians(29), 'Omega': np.radians(144)}
}

# a_b, a_c = 20.7, 33.9
# e_b, e_c = 0.16, 0.042
# i_b, i_c = 130.6, 129.8
# w_b, w_c = 190, 77
# Omega_b, Omega_c = 176, 158
#m_b, m_c, m_d = 0.7, 2.4, 0.4

#%%ä
def simulation(tmax, particle_indices, core_id, a_group, n_planets):

    times = np.linspace(0, tmax, int(tmax*5))

    '''
    Run a rebound simulation for specific particle indices for tmax years
    '''
    start = time.time()

    def create_unique_hash(index, coreid):
        return int(index*1000 + coreid)

    sim = rebound.Simulation()
    sim.units = ['msun', 'yr', 'AU']

    if n_planets == 2:
        star, pb, pc = two_planet_case['star'], two_planet_case['pb'], two_planet_case['pc']

        sim.add(m = star['mass'], x=0, y= 0, z = 0, hash='star', r=R_star)

        sim.add(m = pb['mass'], a = pb['a'], e = pb['e'], f=np.random.rand()*2.*np.pi,
                omega = pb['omega'], inc = pb['inc'], Omega = pb['Omega'], hash = 'pb', r=radii)
        sim.add(m = pc['mass'], a = pc['a'], e = pc['e'], f=np.random.rand()*2.*np.pi,
                omega = pc['omega'], inc = pc['inc'], Omega = pc['Omega'], hash = 'pc', r=radii)
        
    elif n_planets == 3:
        star, pb, pc, pd = three_planet_case['star'], three_planet_case['pb'], three_planet_case['pc'], three_planet_case['pd']

        sim.add(m = star['mass'], x=0, y= 0, z = 0, hash='star', r=R_star)

        sim.add(m = pd['mass'], a = pd['a'], e = pd['e'], f=np.random.rand()*2.*np.pi,
                omega = pd['omega'], inc = pd['inc'], Omega = pd['Omega'], hash = 'pd', r=radii)

        sim.add(m = pb['mass'], a = pb['a'], e = pb['e'], f=np.random.rand()*2.*np.pi,
                omega = pb['omega'], inc = pb['inc'], Omega = pb['Omega'], hash = 'pb', r=radii)
        sim.add(m = pc['mass'], a = pc['a'], e = pc['e'], f=np.random.rand()*2.*np.pi,
                omega = pc['omega'], inc = pc['inc'], Omega = pc['Omega'], hash = 'pc', r=radii)
        
    for i, a in zip(particle_indices, a_group):
        unique_hash = create_unique_hash(i, core_id)
        inc = np.random.uniform(-10, 10)
        sim.add(a=a, inc=np.radians(inc), f=np.random.rand()*2.*np.pi, e=0, 
                primary=sim.particles['star'], hash=unique_hash)
        

    sim.N_active = n_planets + 1
    sim.move_to_com()

    trajectories = {create_unique_hash(i, core_id) : {'ejected': False, 'migrated': False,
                                                      'collided': False, 'captured': False,
                                                    'captured_counter':0, 'captured_t0':None,
                                                      'migrated_peri': False} for i in particle_indices}
    
    filename = f'core_outputs_yr2/core_{core_id}_{tmax}_yr_{N_particles}_{n_planets}_pl_w_captures_and_saving_every_100_yrs_retry.nc'
    #filename = f'core_outputs_yr2/tests/core_{core_id}_test_1.nc'
    with netCDF4.Dataset(filename, 'w') as ncfile:

        sampling_period = 500 #every x/5 = 1000
        s_times = [time for i, time in enumerate(times) if i%sampling_period==0]
        n_saved_times = len(s_times)
        ncfile.createDimension('times_to_save', n_saved_times)
        #ncfile.createDimension('time', len(times))
        ncfile.createDimension('particle', sim.N-1)
        ncfile.createDimension('ejected_p', None) 
        ncfile.createDimension('massive_p', n_planets + 1) #star + planets
        #ncfile.createDimension('saved_parameters', 8)
        ncfile.createDimension('saved_ej_param', 11)
        ncfile.createDimension('saved_collided_p', 12) #track which planet it collides with
        #ncfile.createDimension('migrated_p', None)
        ncfile.createDimension('migrated_peri_p', None)
        ncfile.createDimension('all_saved_times', None)
        ncfile.createDimension('collided_p', None)
        ncfile.createDimension('captured_p', None)

        #times_var = ncfile.createVariable('times', 'f4', ('time',))
        test_particles_var = ncfile.createVariable('test_particles', np.float64, ('times_to_save', 'particle', 'saved_ej_param'))
        massive_bods_var = ncfile.createVariable('massive_bodies', np.float64, ('all_saved_times', 'massive_p', 'saved_ej_param'))
        ejected_var = ncfile.createVariable('ejected', np.float64, ('ejected_p', 'saved_ej_param'))
        #migrated_var = ncfile.createVariable('migrated', 'f4', ('migrated_p', 'saved_ej_param') )
        migrated_peri_var = ncfile.createVariable('migrated_peri', np.float64, ('migrated_peri_p', 'saved_ej_param') )
        collided_var = ncfile.createVariable('collided', np.float64, ('collided_p', 'saved_collided_p'))
        captured_var = ncfile.createVariable('captured', np.float64, ('captured_p', 'saved_collided_p'))
        
        #times_var[:] = times

        ncfile.description = f'simulation results from core {core_id}. Trevascus 2025 values (inc masses). {n_planets} planets!! "peri mig" when R<18. do not delete particles when migrated. tracking captures and collisions. saving every 1000 yrs. this is a redisribution of 1k particles per core to 340 particles per core (17 cores became 50)'
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
                    collided_var[count_collided, :] = [sim.t, pp.x, pp.y, pp.z, pp.e, pp.a, pp.inc, pp.f, pp.Omega, pp.omega, hash_val.value, planet_hash.value]

            return 0
        

        sim.collision = 'line'
        sim.collision_resolve = collision_resolve

        def hill_radius(p):
            return p.a * (1 - p.e) * (p.m / (3 * sim.particles['star'].m))**(1/3)
        
        hill_pb = hill_radius(sim.particles['pb'])
        hill_pc = hill_radius(sim.particles['pc'])

        if n_planets == 3:
            hill_pd = hill_radius(sim.particles['pd'])


        for t, tid in enumerate(times):
            
            if t > 0:
                dt = times[t] - times[t-1]
                sim.integrate(sim.t + dt)
                hill_pb = hill_radius(sim.particles['pb']) #update 
                hill_pc = hill_radius(sim.particles['pc']) #update 
            
                if n_planets == 3:
                    hill_pd = hill_radius(sim.particles['pd']) #update

            apo_dist = np.array([sim.particles['pb'].a * (1+sim.particles['pb'].e),
                                sim.particles['pc'].a * (1+sim.particles['pc'].e)])
            peri_dist = np.array([sim.particles['pb'].a * (1-sim.particles['pb'].e),
                                sim.particles['pc'].a * (1-sim.particles['pc'].e)])
            eccs = np.array([sim.particles['pb'].e, sim.particles['pc'].e])


            c1 = np.all(eccs < 1)
            c2 = apo_dist[0] < peri_dist[1] 

            if n_planets == 3:
                apo_d = sim.particles['pd'].a * (1+sim.particles['pd'].e)
                c2 = c2 and apo_d < peri_dist[0]

            if not (c1 and c2):
                with open(f'progress_tracking_files/failed_cores_{n_planets}pl_outerdisc.txt', 'a') as f:
                    f.write(f'File {filename} failed at and t = {tid} yr.\n')

                print(f'Core {core_id} failed.')
                return None
            
            
            to_remove = []

            for p in sim.particles[sim.N_active:]:
                h = p.hash.value
                traj = trajectories[h]

                dist = np.sqrt(p.x**2 + p.y**2 + p.z**2)
        
                if p.e >= 1 and dist > 200 and not traj['ejected']:
                    traj['ejected'] = True
                    ejected_var[count_ejected, :] = [tid, p.x, p.y, p.z, p.e, p.a, p.inc, p.f, p.Omega, p.omega, h] 
                    to_remove.append(h)
                    count_ejected += 1

                if dist <= 18 and p.e < 1 and not traj['migrated_peri']:
                    traj['migrated_peri'] = True
                    migrated_peri_var[count_migrated_peri,:] = [tid, p.x, p.y, p.z, p.e, p.a, p.inc, p.f, p.Omega, p.omega, p.hash.value]
                    
                    if n_planets == 3:
                        massive_bods = ['star', 'pb', 'pc', 'pd']
                    elif n_planets == 2:
                        massive_bods = ['star', 'pb', 'pc']

                    for j, plhash in enumerate(massive_bods):
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
                inside_pb = d_pb < hill_pb

                d_pc = np.sqrt((p.x - sim.particles['pc'].x)**2 +
                                (p.y - sim.particles['pc'].y)**2 +
                                (p.z - sim.particles['pc'].z)**2)
                inside_pc = d_pc < hill_pc

                capturing_planet = None

                if inside_pb:
                    capturing_planet = sim.particles['pb']
                elif inside_pc:
                    capturing_planet = sim.particles['pc']


                if n_planets == 3:
                    d_pd = np.sqrt((p.x - sim.particles['pd'].x)**2 +
                                    (p.y - sim.particles['pd'].y)**2 +
                                    (p.z - sim.particles['pd'].z)**2)
                    inside_pd = d_pd < hill_pd

                    if inside_pd:
                        capturing_planet = sim.particles['pd']
                        
                        if traj['captured_t0'] is None:
                            traj['captured_t0'] = sim.t
                        traj['captured_counter'] +=1
                    
                    else:
                        traj['captured_counter'] = 0
                        traj['captured_t0'] = None

                if capturing_planet is not None:
                    if traj['captured_t0'] is None:
                        traj['captured_t0'] = sim.t
                    traj['captured_counter'] +=1

                else: #reset if not actually captured
                    traj['captured_counter'] = 0
                    traj['captured_t0'] = None

                if not traj['captured'] and not traj['ejected']:
                    if (traj['captured_counter'] > 50000): # 10 thousand years
                        print(f'captured by {capturing_planet.hash.value}', 'r:', dist)
                        traj['captured'] = True
                        captured_var[count_captured, :] = [tid, p.x, p.y, p.z, p.e, p.a, p.inc, p.f, p.Omega, p.omega, h, capturing_planet.hash.value]
                        to_remove.append(h)
                        count_captured += 1

                
                if traj['collided']:
                    print('collided')
                    to_remove.append(h)


            for h in to_remove:
                sim.remove(hash=h)
                print(f'Particle {h} removed from simulation.')

            if t % sampling_period == 0:
                for i, p in enumerate(sim.particles[1:]):
                    test_particles_var[save_t_index, i, :] = [tid, p.x, p.y, p.z, p.e, p.a, p.inc, p.f, p.Omega, p.omega, p.hash.value]
                save_t_index += 1
      
    end = time.time()

    print(f'core {core_id} done')
    with open(f'progress_tracking_files/progress_{tmax}_yr_{n_planets}pl_outerdisc.txt', 'a') as f:
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
    
def prompt_n_planets():
    while True:
        try:
            n_pl = int(input("How many planets do you want to simulate? (2 or 3): "))
            if n_pl in (2, 3):
                return n_pl
            else:
                print("Please enter 2 or 3.")
        except ValueError:
            print("no silly, I said 2 or 3.")
    
    
def parallelization(N_testparticles, tmax, N_cores, n_planets):

    with open(f'progress_tracking_files/progress_{tmax}_yr_{n_planets}pl_outerdisc.txt', 'w') as f:
        pass

    with open(f'progress_tracking_files/failed_cores_{n_planets}pl_outerdisc.txt', 'w') as f:
        pass

    a_vals = np.random.uniform(54, 87, N_testparticles)

    indices = np.arange(N_testparticles)
    sorted_idx = np.argsort(a_vals)   # indices of particles sorted by semimajor axis
    a_vals = a_vals[sorted_idx]       # sorted semimajor axes
    indices = indices[sorted_idx]   

    groups = np.array_split(indices, N_cores)
    a_groups = np.array_split(a_vals, N_cores)

    failed_cores = [17, 19, 28, 30, 31, 32, 33, 38, 39, 40, 41, 43, 44, 45, 46, 47, 49]

    failed_groups = [groups[i] for i in failed_cores]
    failed_a_groups = [a_groups[i] for i in failed_cores]
    failed_indices = np.concatenate(failed_groups)
    failed_a_vals  = np.concatenate(failed_a_groups)

    N_new_cores = 50

    new_groups = np.array_split(failed_indices, N_new_cores)
    new_a_groups = np.array_split(failed_a_vals, N_new_cores)
    print(new_a_groups)

    print(f"Rerunning {len(failed_indices)} particles across {N_new_cores} cores")


    core_offset = 100

    chunk_results = Parallel(n_jobs=N_new_cores)(
        delayed(simulation)(tmax, new_groups[i], i+core_offset, new_a_groups[i], n_planets)
        for i in range(N_new_cores))

    
    return chunk_results

#%%
from notify_run import Notify

notify = Notify()
print(notify.endpoint)
if __name__ == '__main__':

    N_particles = 50000
    tmax = 5e6
    N_cores = 50


    # notify.send('Script started')

    if prompt():
        n_planets = prompt_n_planets()      

        filenames = parallelization(N_particles, tmax, N_cores, n_planets)
        
    else:
        print('I think you should change your file name first. You are welcome.')

    # notify.send('Script done! ✅')

