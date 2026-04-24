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
import reboundx

# Limit to 100 GB of virtual memory
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
# a_b, a_c = 20.7, 33.9
# e_b, e_c = 0.16, 0.042
# i_b, i_c = 130.6, 129.8
# w_b, w_c = 190, 77
# Omega_b, Omega_c = 176, 158

dtor = np.pi / 180
#%%ä

def simulation(tmax, particle_seed, core_id, n_planets):
    times = np.linspace(0, tmax, int(tmax*5))
    '''
    Run a rebound simulation for specific particle indices for tmax years
    '''
    start = time.time()

    def create_unique_hash(index, coreid):
        return int(index*234 + coreid)

    sim = rebound.Simulation()
    sim.units = ['msun', 'yr', 'AU']

    p = particle_seed['particle']
    pb = particle_seed['pb']
    pc = particle_seed['pc']
    star = particle_seed['star']

    sim.add(m = 0.952, x=star['x'], y=star['y'], z=star['z'], hash='star', r = R_star)

    if n_planets == 3:
        pd = particle_seed['pd']
        m_b, m_c, m_d = 0.7, 2.4, 0.4
        sim.add(m = m_d*jtos, a = pd['a'], e = pd['e'], f=pd['f'],
            inc =pd['inc'], omega = pd['omega'], Omega =  pd['Omega'], hash='pd')

    if n_planets == 2:
        m_b, m_c = 1.4, 6.4

    sim.add(m = m_b*jtos, a = pb['a'], e = pb['e'], f=pb['f'], primary=sim.particles['star'],
            omega = pb['omega'], inc = pb['inc'], Omega = pb['Omega'], hash = 'pb', r=R_b)

    sim.add(m = m_c*jtos, a = pc['a'], e = pc['e'], f=pb['f'], primary=sim.particles['star'],
            omega = pc['omega'], inc = pc['inc'], Omega = pc['Omega'], hash = 'pc', r=R_c)
    
    # for i, a in zip(particle_indices, a_group):
    #     unique_hash = create_unique_hash(i, core_id)
    #     inc = np.random.uniform(-10, 10)
    #     sim.add(a=a, inc=np.radians(inc), f=np.random.rand()*2.*np.pi, e=0, r=6.68e-9, hash=unique_hash)

    sim.add(m=0, e=p['e'],
            a=p['a'], inc=p['inc'], f=p['f'], hash=int(p['hash']), primary=sim.particles['star'], r=6.68e-9)
    
    def perturb(value, frac=0.01):
        return value * (1 + np.random.uniform(-frac, frac))
    

    for i in range(10):
        sim.add(m=0, e=perturb(p['e']),
                a=perturb(p['a']), inc=perturb(p['inc']), f=perturb(p['f']), hash=create_unique_hash(i, core_id), r=6.68e-9)
    
    sim.N_active = n_planets + 1
    sim.move_to_com()

    n_test_particles = len(sim.particles) - sim.N_active

    testp_hashes = [p.hash.value for p in sim.particles[sim.N_active:]]

    rebx = reboundx.Extras(sim)

    myforce = rebx.load_force("test_force")     
    rebx.add_force(myforce)

    trajectories = {h : {'ejected': False, 'collided': False, 'captured': False, 'star_grazed': False,
                        'captured_counter':0, 'captured_t0':None} for h in testp_hashes}
    
    filename = f'core_outputs_yr2/gas_drag/core_{core_id}_{tmax}_yr_ptcl_{particle_seed["particle"]["hash"]}_{n_planets}_pl_w_planet_params.nc'
    #filename = f'core_outputs_yr2/test_{core_id}.nc'
    with netCDF4.Dataset(filename, 'w') as ncfile:

        #saving every 10 years
        sampling_period = 50
        s_times = [time for i, time in enumerate(times) if i%sampling_period==0]
        n_saved_times = len(s_times)
        ncfile.createDimension('times_to_save', n_saved_times)
        ncfile.createDimension('time', len(times))
        ncfile.createDimension('particle', n_test_particles)
        ncfile.createDimension('ejected_p', None) 
        ncfile.createDimension('massive_p', n_planets+1)
        ncfile.createDimension('saved_ej_param', 11)
        ncfile.createDimension('all_saved_times', None)
        ncfile.createDimension('saved_collided_p', 12) #track which planet it collides with/ is captured by
        ncfile.createDimension('collided_p', None)
        ncfile.createDimension('captured_p', None)

        times_var = ncfile.createVariable('times', 'f4', ('time',))
        test_particles_var = ncfile.createVariable('test_particles', 'f4', ('times_to_save', 'particle', 'saved_ej_param'))
        massive_bods_var = ncfile.createVariable('massive_bodies', 'f4', ('all_saved_times', 'massive_p', 'saved_ej_param'))
        ejected_var = ncfile.createVariable('ejected', 'f4', ('ejected_p', 'saved_ej_param'))
        collided_var = ncfile.createVariable('collided', 'f4', ('collided_p', 'saved_collided_p'))
        captured_var = ncfile.createVariable('captured', 'f4', ('captured_p', 'saved_collided_p'))
        
        times_var[:] = times

        ncfile.description = f'simulation results from core {core_id}. Trevascus 2025 values (inc masses). {n_planets} planets!! inner disc, cluster of particles. deleting collision particles. gas drag OFF. this one is better bc using omega and Omega for planets.'
        ncfile.history = 'created' + time.ctime(time.time())

        count_ejected = 0
        count_collided = 0
        count_captured = 0
        save_t_index = 0
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
                planet_hash = h1 if sim.particles[h2].m == 0 else h2

                if not trajectories[hash_val.value]['collided']:
                    trajectories[hash_val.value]['collided'] = True
                    print(f'collided! {p1.hash, p2.hash}')
                    # Save parameters in NetCDF
                    pp = sim.particles[hash_val]
                    collided_var[count_collided, :] = [sim.t, pp.x, pp.y, pp.z, pp.e, pp.a, pp.inc, pp.f, pp.Omega, pp.omega, pp.hash.value, planet_hash.value]
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

            to_remove.clear()   

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
            c2 = (apo_dist[0] < peri_dist[1])
            
            if n_planets == 3:
                apo_d = sim.particles['pd'].a * (1+sim.particles['pd'].e)
                c2 = c2 and apo_d < peri_dist[0]

            if not (c1 and c2):
                with open(f'progress_tracking_files/failed_cores_{n_pl}.txt', 'a') as f:
                    f.write(f'File {filename} failed at and t = {tid} yr.\n')
                print(f'Core {core_id} failed.')
                return None

            if t % sampling_period == 0:
                
                if n_planets == 3:
                    massive_bods = ['star', 'pb', 'pc', 'pd']
                else:
                    massive_bods = ['star', 'pb', 'pc']

                for j, hash in enumerate(massive_bods):
                    p = sim.particles[hash]
                    if hash != 'star':
                        massive_bods_var[save_t_index, j, :] = [t, p.x, p.y, p.z, p.e, p.a, p.inc, p.f, p.Omega, p.omega, 22222]
                    else:
                        massive_bods_var[save_t_index, j, :] = [t, p.x, p.y, p.z, 0, 0, 0, 0, 0, 0, 11111]
            
                for i, p in enumerate(sim.particles[sim.N_active:]):
                    test_particles_var[save_t_index, i, :] = [tid, p.x, p.y, p.z, p.e, p.a, p.inc, p.f, p.Omega, p.omega, p.hash.value]

                save_t_index += 1

            for pt in sim.particles[sim.N_active:]:
                h = pt.hash.value
                traj = trajectories[h]
                dist = np.sqrt(pt.x**2 + pt.y**2 + pt.z**2)

                if pt.a < 0 and dist > 200 and not traj['ejected']:
                    print(f'{h}: ejected')
                    traj['ejected'] = True
                    ejected_var[count_ejected, :] = [tid, pt.x, pt.y, pt.z, pt.e, pt.a, pt.inc, pt.f, pt.Omega, pt.omega, h] 
                    to_remove.append(h)
                    count_ejected += 1  

                if dist <= 1 and not traj['star_grazed']:
                    print(f'{h}: star grazed')
                    traj['star_grazed'] = True
                    to_remove.append(h) 

                d_pb = np.sqrt((pt.x - sim.particles['pb'].x)**2 +
                (pt.y - sim.particles['pb'].y)**2 +
                (pt.z - sim.particles['pb'].z)**2)

                d_pc = np.sqrt((pt.x - sim.particles['pc'].x)**2 +
                                (pt.y - sim.particles['pc'].y)**2 +
                                (pt.z - sim.particles['pc'].z)**2)

                capturing_planet = None

                if n_planets == 3:
                    d_pd = np.sqrt((pt.x - sim.particles['pd'].x)**2 +
                                    (pt.y - sim.particles['pd'].y)**2 +
                                    (pt.z - sim.particles['pd'].z)**2)
                    inside_pd = d_pd < hill_pd

                    if inside_pd:
                        capturing_planet = sim.particles['pd']

                inside_pb = d_pb < hill_pb
                inside_pc = d_pc < hill_pc

                if inside_pb:
                    capturing_planet = sim.particles['pb']
                if inside_pc:
                    capturing_planet = sim.particles['pc']

                if capturing_planet is not None:
                    if traj['captured_t0'] is None:
                        traj['captured_t0'] = sim.t
                    traj['captured_counter'] +=1

                else: #reset if not actually captured
                    traj['captured_counter'] = 0
                    traj['captured_t0'] = None

                if not traj['captured'] and not traj['ejected'] and pt.a > 0:
                    if (traj['captured_counter'] > 50000): #50 thousand years
                        print(f'captured {h}, planet is {capturing_planet.hash.value}')
                        traj['captured'] = True
                        captured_var[count_captured, :] = [tid, pt.x, pt.y, pt.z, pt.e, pt.a, pt.inc, pt.f, pt.Omega, pt.omega, h, capturing_planet.hash.value]
                        to_remove.append(h)
                        count_captured += 1

                
                if traj['collided'] and not traj['captured'] and not traj['ejected'] and not traj['star_grazed']:
                    print(f'{h}: collided')
                    to_remove.append(h)
                    count_collided += 1


            for h in to_remove:
                sim.remove(hash=h)

        print(f'At {sim.t} yr:', len(sim.particles)-sim.N_active, 'particles')
        

    end = time.time()

    print(f'core {core_id} done')
    with open(f'progress_tracking_files/progress_{tmax}_yr_{n_pl}_gas_drag_on.txt', 'a') as f:
        f.write(f'Core {core_id} finished and took {(end-start)/3600:.3f} h.\n')

    return filename

#test this function
#%%

'''getting parameters for the test particles'''

def find_mig_and_ej(filename, n_planets):
    global n_particles

    mig_peri_list = []

    N = len(filename)
    print(f'Initially: {N} files')

    filecount = 0

    for index, file in enumerate(filename):
        with netCDF4.Dataset(file, 'r') as ncfile:
            massive_bodies_a = ncfile['test_particles'][-1, 0:n_planets, 5]

            massive_bodies_e = ncfile['test_particles'][-1, 0:n_planets, 4]
            dist = massive_bodies_a * (1+massive_bodies_e)
            dist_small = massive_bodies_a * ( 1 - massive_bodies_e)


            c1 = np.all(massive_bodies_e < 1)

            if n_planets == 2:
                c3 = ((dist[0] < dist_small[1]))
            if n_planets == 3:
                # note that test particles are ordered from closest to furthest out!!
                c3 = ((dist[0] < dist_small[1])) and (dist[1] < dist_small[2]) 
                
            if c1 and c3:    
                filecount += 1

                migrated_peri = ncfile['migrated_peri'][:]
                massive_bodies = ncfile['massive_bodies'][:]

                if np.any(migrated_peri):

                    for i in range(migrated_peri.shape[0]):
                        unmasked_data_peri = migrated_peri[i].compressed()

                        if unmasked_data_peri.size > 0:
                            
                            tid = unmasked_data_peri[0]

                            t_idx = np.where(massive_bodies[:, 0, 0] == tid)[0][0]

                            star = massive_bodies[t_idx, 0, :]
                            pb = massive_bodies[t_idx, 1, :]
                            pc = massive_bodies[t_idx, 2, :]
                            
                            if n_planets == 3:
                                pd = massive_bodies[t_idx, 3, :]


                            entry = {
                                "particle": {
                                    "tid": float(tid),
                                    "e": float(unmasked_data_peri[4]),
                                    "a": float(unmasked_data_peri[5]),
                                    "inc": float(unmasked_data_peri[6]),
                                    "f": float(unmasked_data_peri[7]),
                                    "Omega": float(unmasked_data_peri[8]),
                                    "omega": float(unmasked_data_peri[9]),
                                    "hash": int(unmasked_data_peri[10]),
                                },
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
                                    "x": float(star[1]),
                                    "y": float(star[2]),
                                    "z": float(star[3]),
                                }
                            }

                            if n_planets == 3:
                                entry["pd"] = {
                                    "e": float(pd[4]),
                                    "a": float(pd[5]),
                                    "inc": float(pd[6]),
                                    "f": float(pd[7]),
                                    "Omega": float(pd[8]),
                                    "omega": float(pd[9]),
                                }

                            mig_peri_list.append(entry)

                                
    n_particles = 200 * filecount

    print(f'Finally: {filecount} files => {n_particles} particles')

    results = {
        'mig_all': mig_peri_list,
        'n_particles': n_particles, #should be 8400 for this simulation 
    }

    #np.savez('hist_data_w_all_parameters.npz', **results)
    return results #{'migrated': migrated_array, 'ejected': ejected_array}

files = glob.glob(f'core_outputs_yr2/*3_pl*saving*')
results = find_mig_and_ej(files, 3)
mig_all = results['mig_all']

print(mig_all['pd']['inc'])
#%%

def prompt():
    confirmation = input('Ally, did you remember to change the name of your file and update file description? And starting index? Type "YES" if so.')
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
            print("Please enter a valid integer (2 or 3).")
    
def parallelization(start_idx, N_particles, tmax, N_cores, mig_all, n_pl):

    # with open(f'progress_tracking_files/progress_{tmax}_yr_gas_drag_on.txt', 'w') as f:
    #     pass

    print(f'You have {len(mig_all)} particles to choose from.')

    end_idx = start_idx + N_particles 
    seeds = mig_all[start_idx:end_idx]

    chunk_results = Parallel(n_jobs=N_cores)(
        delayed(simulation)(tmax, ptcl, core_id, n_pl)
        for core_id, ptcl in enumerate(seeds, start=start_idx)
    )
        
    return chunk_results

#%%
if __name__ == '__main__':

    tmax = 1e6
    N_cores = 50
    N_particles = N_cores
    start_idx = 0

    
    two_pl_files = glob.glob(f'core_outputs_yr2/*2_pl*saving*.nc')
    three_pl_files = glob.glob(f'core_outputs_yr2/*3_pl_*saving_every_100_yrs.nc')

    if prompt():                       # ← your existing prompt
        n_pl = prompt_n_planets()      # ← new prompt

        if n_pl == 2:
            files = two_pl_files
        else:
            files = three_pl_files

        mig_all = find_mig_and_ej(files, n_pl)['mig_all']
        filenames = parallelization(
            start_idx, N_particles, tmax, N_cores, mig_all, n_pl
        )

    else:
        print('I think you should change your file name first. You are welcome.')