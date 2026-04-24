#%%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import rebound
import astropy.constants as const
from multiprocessing import Pool
from tqdm import tqdm 
from joblib import Parallel, delayed
# import os 
import time
import netCDF4
import glob
import resource
import reboundx

# Limit to 100 GB of virtual memory
limit = 100 * 1024 ** 3
resource.setrlimit(resource.RLIMIT_AS, (limit, limit))

jtos = const.M_jup / const.M_sun

# R_b = 2.72 * const.R_jup.to('au').value
# R_c = 2.04 * const.R_jup.to('au').value

R_b = 10 * const.R_jup.to('au').value
R_c = 10 * const.R_jup.to('au').value
R_d = 10 * const.R_jup.to('au').value
R_star = 1.26 * const.R_sun.to('au').value

dtor = np.pi / 180

#%%ä

def simulation(tmax, particle_seed, core_id, n_planets):


    '''
    Run a rebound simulation for specific particle indices for tmax years
    '''
    start = time.time()

    sim = rebound.Simulation()
    sim.units = ['msun', 'yr', 'AU']

    p = particle_seed['particle']
    pb = particle_seed['pb']
    pc = particle_seed['pc']
    star = particle_seed['star']

    t_start = particle_seed['particle']['tid'][0]
    times = np.linspace(t_start, tmax, int(tmax*5)) #every 0.2 yrs

    dt = 1.0 / 5 

    if n_planets == 2:
        sim.add(m = 0.952, x=star['x'], y=star['y'], z=star['z'], hash='star', r = R_star)
        m_b, m_c = 1.4, 6.4

    sim.add(m = m_b*jtos, a = pb['a'], e = pb['e'], f=pb['f'],
            omega = pb['omega'], inc = pb['inc'], Omega = pb['Omega'], hash = 'pb', r=R_b)

    sim.add(m = m_c*jtos, a = pc['a'], e = pc['e'], f=pc['f'], 
            omega = pc['omega'], inc = pc['inc'], Omega = pc['Omega'], hash = 'pc', r=R_c)

    print(f'Added planets, which are at {sim.orbits()}')
    N = len(p["a"])

    for i in range(N):
        sim.add(
            m=0.0,  # test particles
            a=p["a"][i],
            e=p["e"][i],
            inc=p["inc"][i],
            Omega=p["Omega"][i],
            omega=p["omega"][i],
            f=p["f"][i],
            hash=int(p["hash"][i])
        )


    sim.N_active = n_planets + 1
    sim.move_to_com()

    trajectories = {h : {'ejected': False, 'migrated': False,
                                                      'collided': False, 'captured': False,
                                                    'captured_counter':0, 'captured_t0':None,
                                                      'migrated_peri': False} for h in p['hash']}
    
    filename = f'core_outputs_yr2/core_{core_id}_{tmax}_yr_50000_{n_planets}_pl_w_captures_and_saving_every_100_yrs_RESUMED.nc'
    
    #filename = f'core_outputs_yr2/tests/core_{core_id}_test_2.nc'
    with netCDF4.Dataset(filename, 'w') as ncfile:

        sampling_period = 500 #every x/5 = 100
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

        ncfile.description = f'these are a continuation of {core_id}. Trevascus 2025 values (inc masses). {n_planets} planets!! "peri mig" when R<18. do not delete particles when migrated. tracking captures and collisions. saving every 100 yrs. starting from the last saved time.'
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
                    

                    print(f'collided! {p1.hash, p2.hash}')
                    # Save parameters in NetCDF
                    pp = sim.particles[hash_val]
                    collided_var[count_collided, :] = [sim.t, pp.x, pp.y, pp.z, pp.e, pp.a, pp.inc, pp.f, pp.Omega, pp.omega, hash_val.value, planet_hash.value]
                    count_collided += 1
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

                print(f'Core {core_id} failed because orbits crossed. b is at {apo_dist[0]} and c is at {peri_dist[1]}')
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
    with open(f'progress_tracking_files/progress_{tmax}_yr_{n_pl}_outerdisc_restart.txt', 'a') as f:
        f.write(f'Core {core_id} finished and took {(end-start)/3600:.3f} h.\n')

    return filename

#test this function
#%%

'''getting parameters for the test particles'''

def get_all_deets(filename):
    global n_particles

    entries = []
    N = len(filename)
    print(f'{N} files')

    for index, file in enumerate(filename):
        with netCDF4.Dataset(file, 'r') as ncfile:

            t_max = np.max(ncfile['test_particles'][:, :, 0 ])
            print(t_max)
            t_max_arg = int(t_max / 100)
            all_particle_info = ncfile['test_particles'][t_max_arg, :, :]
            
            all_particle_info = all_particle_info.filled(np.nan)

            valid_mask = np.all(np.isfinite(all_particle_info[:, 4:11]), axis=1)
            new_valid_info = all_particle_info[valid_mask]
            print(new_valid_info.shape)

            pb, pc = all_particle_info[0], all_particle_info[1]
            #print(all_particle_info[4, -1].type)
            
            m_star = 0.952
            m_b, m_c = 1.4*jtos, 6.4*jtos
            
            r_b = np.array([pb[1], pb[2], pb[3]])
            r_c = np.array([pc[1], pc[2], pc[3]])

            star = -(m_b * r_b + m_c * r_c) / m_star

            particles = {
                "tid": new_valid_info[:, 0].astype(float),
                "e": new_valid_info[:, 4].astype(float),
                "a": new_valid_info[:, 5].astype(float),
                "inc": new_valid_info[:, 6].astype(float),
                "f": new_valid_info[:, 7].astype(float),
                "Omega": new_valid_info[:, 8].astype(float),
                "omega": new_valid_info[:, 9].astype(float),
                "hash": new_valid_info[:, 10].astype(int),
            }

            entry = {
                "particle": particles,
                "pb": {
                    "e": float(pb[4]),
                    "a": float(pb[5]),
                    "inc": float(pb[6]),
                    "f": float(pb[7]),
                    "Omega": float(pb[8]),
                    "omega": float(pb[9]),
                },
                "pc": {
                    "e": float(pc[4]),
                    "a": float(pc[5]),
                    "inc": float(pc[6]),
                    "f": float(pc[7]),
                    "Omega": float(pc[8]),
                    "omega": float(pc[9]),
                },
                "star": {
                    "x": float(star[0]),
                    "y": float(star[1]),
                    "z": float(star[2]),
                }
            }

                        # if n_planets == 3:
                        #     entry["pd"] = {
                        #         "e": float(pd[4]),
                        #         "a": float(pd[5]),
                        #         "inc": float(pd[6]),
                        #         "f": float(pd[7]),
                        #         "Omega": float(pd[8]),
                        #         "omega": float(pd[9]),
                        #     }

                        # mig_peri_list.append(entry)

                                
            entries.append(entry)

    return entries

files = [f'core_outputs_yr2/core_{i}_5000000.0_yr_50000_2_pl_w_captures_and_saving_every_100_yrs.nc' for i in [34, 35, 36, 42, 48]]
#%%

def prompt():
    confirmation = input('Ally, did you remember to change the name of your file and update file description? Type "YES" if so.')
    if confirmation.upper() == 'YES':
        print('Okay, I believe you, I will now simulate your little planets.')
        return True
    else:
        return False
    
  
def parallelization(tmax, N_cores, files, n_pl):

    with open(f'progress_tracking_files/progress_{tmax}_yr_{n_pl}_outerdisc_restart.txt', 'w') as f:
        pass

    with open(f'progress_tracking_files/failed_cores_{n_pl}_restart.txt', 'w') as f:
        pass

    chunk_results = Parallel(n_jobs=N_cores, batch_size=1)(
        delayed(simulation)(tmax, particle, core, n_pl)
        for (particle, core) in zip(files, [34, 35, 36, 42, 48]))
    
        
    return chunk_results

#%%
if __name__ == '__main__':

    tmax = 5e6

    files = [f'core_outputs_yr2/core_{i}_5000000.0_yr_50000_2_pl_w_captures_and_saving_every_100_yrs.nc' for i in [34, 35, 36, 42, 48]]
    N_cores = len(files)

    if prompt():                       # ← your existing prompt
        n_pl = 2

        particles = get_all_deets(files)
        print(len(particles))
        
        filenames = parallelization(
           tmax, N_cores, particles, n_pl
        )

    else:
        print('I think you should change your file name first. You are welcome.')