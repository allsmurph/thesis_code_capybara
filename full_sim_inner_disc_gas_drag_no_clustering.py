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

print(reboundx.__file__)
print(reboundx.__version__)

print(rebound.__file__)
print(rebound.__version__)


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

def simulation(tmax, particle_seed, core_id, n_planets, pl_size):

    times = np.linspace(0, tmax, int(tmax*5))
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

    if n_planets == 3:
        sim.add(m = 0.965, x=star['x'], y=star['y'], z=star['z'], hash='star', r = R_star)

        pd = particle_seed['pd']
        m_b, m_c, m_d = 0.7, 2.4, 0.4

        sim.add(m = m_d*jtos, a = pd['a'], e = pd['e'], f=pd['f'],
            inc = pd['inc'], omega = pd['omega'], Omega = pd['Omega'], hash='pd', r=R_d)

    if n_planets == 2:
        sim.add(m = 0.952, x=star['x'], y=star['y'], z=star['z'], hash='star', r = R_star)
        m_b, m_c = 1.4, 6.4

    sim.add(m = m_b*jtos, a = pb['a'], e = pb['e'], f=pb['f'],
            omega = pb['omega'], inc = pb['inc'], Omega = pb['Omega'], hash = 'pb', r=R_b)

    sim.add(m = m_c*jtos, a = pc['a'], e = pc['e'], f=pc['f'], 
            omega = pc['omega'], inc = pc['inc'], Omega = pc['Omega'], hash = 'pc', r=R_c)

    sim.add(m=0, e=p['e'], a=p['a'], inc=p['inc'], f=p['f'],
            omega=p['omega'], Omega = p['Omega'], hash=int(p['hash']), r=6.68e-9)

    sim.N_active = n_planets + 1
    sim.move_to_com()

    testp_hashes = [p.hash.value for p in sim.particles[sim.N_active:]]

    rebx = reboundx.Extras(sim)

    if pl_size == 100:
        myforce = rebx.load_force("gas_drag_100km")  

    elif pl_size == 10:
        myforce = rebx.load_force("gas_drag_10km")

    elif pl_size == 1:
        myforce = rebx.load_force("gas_drag_1km")   
    
    rebx.add_force(myforce)

    trajectories = {h : {'ejected': False, 'collided': False, 'captured': False, 'star_grazed': False,
                        'captured_counter':0, 'captured_t0':None} for h in testp_hashes}
    
    filename = f'core_outputs_yr2/gas_drag_final/core_{core_id}_{tmax}_yr_ptcl_{particle_seed["particle"]["hash"]}_{n_planets}_pl_single_particle_{pl_size}km.nc'
    #filename = f'core_outputs_yr2/test_{core_id}.nc'
    with netCDF4.Dataset(filename, 'w') as ncfile:

        #saving every 10 years
        sampling_period = 50
        s_times = [time for i, time in enumerate(times) if i%sampling_period==0]
        n_saved_times = len(s_times)
        ncfile.createDimension('times_to_save', n_saved_times)
        ncfile.createDimension('time', len(times))
        ncfile.createDimension('ejected_p', None) 
        ncfile.createDimension('massive_p', n_planets+1)
        ncfile.createDimension('saved_ej_param', 11)
        ncfile.createDimension('all_saved_times', None)
        ncfile.createDimension('saved_collided_p', 12) #track which planet it collides with/ is captured by
        ncfile.createDimension('collided_p', None)
        ncfile.createDimension('captured_p', None)
        ncfile.createDimension('grazed_p', None)

        times_var = ncfile.createVariable('times', np.float64, ('time',))
        test_particles_var = ncfile.createVariable('test_particles', np.float64, ('times_to_save', 'saved_ej_param'))
        massive_bods_var = ncfile.createVariable('massive_bodies', np.float64, ('all_saved_times', 'massive_p', 'saved_ej_param'))
        ejected_var = ncfile.createVariable('ejected', np.float64, ('ejected_p', 'saved_ej_param'))
        collided_var = ncfile.createVariable('collided', np.float64, ('collided_p', 'saved_collided_p'))
        captured_var = ncfile.createVariable('captured', np.float64, ('captured_p', 'saved_collided_p'))
        star_grazed_var =  ncfile.createVariable('star_grazed', np.float64, ('grazed_p', 'saved_ej_param'))
        times_var[:] = times

        ncfile.description = f'simulation results from core {core_id}. Trevascus 2025 values (inc masses). {n_planets} planets!! inner disc, 1 particle, i.e., no clustering. deleting collision particles. gas drag on. using omega and Omega for planets. star grazed when peri < 0.4 au, fixed stellar masses.'
        ncfile.history = 'created' + time.ctime(time.time())

        count_ejected = 0
        count_collided = 0
        count_captured = 0
        count_sg = 0
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

                if not trajectories[hash_val.value]['collided'] and hash_val.value not in to_remove:
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
                if n_planets == 3:
                    massive_bods = ['star', 'pb', 'pc', 'pd']
                elif n_planets == 2:
                    massive_bods = ['star', 'pb', 'pc']

                for j, hash in enumerate(massive_bods):
                    p = sim.particles[hash]
                    if hash != 'star':
                        massive_bods_var[save_t_index, j, :] = [t, p.x, p.y, p.z, p.e, p.a, p.inc, p.f, p.Omega, p.omega, 22222]
                    else:
                        massive_bods_var[save_t_index, j, :] = [t, p.x, p.y, p.z, 0, 0, 0, 0, 0, 0, 11111]
            
                with open(f'progress_tracking_files/failed_cores_{n_pl}_gas_drag_{pl_size}km.txt', 'a') as f:
                    f.write(f'File {filename} failed at and t = {tid} yr.\n')
                print(f'Core {core_id} failed.')
                return None

            if t % sampling_period == 0:
    
                if n_planets == 3:
                    massive_bods = ['star', 'pb', 'pc', 'pd']
                elif n_planets == 2:
                    massive_bods = ['star', 'pb', 'pc']

                for j, hash in enumerate(massive_bods):
                    p = sim.particles[hash]
                    if hash != 'star':
                        massive_bods_var[save_t_index, j, :] = [t, p.x, p.y, p.z, p.e, p.a, p.inc, p.f, p.Omega, p.omega, 22222]
                    else:
                        massive_bods_var[save_t_index, j, :] = [t, p.x, p.y, p.z, 0, 0, 0, 0, 0, 0, 11111]
            
                p = sim.particles[-1]
                test_particles_var[save_t_index, :] = [tid, p.x, p.y, p.z, p.e, p.a, p.inc, p.f, p.Omega, p.omega, p.hash.value]

                save_t_index += 1
 
            pt = sim.particles[-1]
            h = pt.hash.value
            traj = trajectories[h]
            dist = np.sqrt(pt.x**2 + pt.y**2 + pt.z**2)
            peri_dist = pt.a * (1 - pt.e)

            if pt.a < 0 and dist > 200 and not traj['ejected']:
                print(f'{h}: ejected')
                traj['ejected'] = True
                ejected_var[count_ejected, :] = [tid, pt.x, pt.y, pt.z, pt.e, pt.a, pt.inc, pt.f, pt.Omega, pt.omega, h] 
                to_remove.append(h)
                count_ejected += 1  

            #5 is close enough
            if peri_dist <= 0.4 and dist < 5 and not traj['star_grazed'] and h not in to_remove and pt.e < 1:
                print(f'{h}: star grazed')
                traj['star_grazed'] = True
                star_grazed_var[count_sg, :] = [tid, pt.x, pt.y, pt.z, pt.e, pt.a, pt.inc, pt.f, pt.Omega, pt.omega, h] 
                #to_remove.append(h) 
                count_sg += 1

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
                    
                else:
                    traj['captured_counter'] = 0
                    traj['captured_t0'] = None

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

            if not traj['captured'] and not traj['ejected'] and h not in to_remove:
                if (traj['captured_counter'] > 50000): #10 thousand years
                    print(f'captured {h}, planet is {capturing_planet.hash.value}')
                    traj['captured'] = True
                    captured_var[count_captured, :] = [tid, pt.x, pt.y, pt.z, pt.e, pt.a, pt.inc, pt.f, pt.Omega, pt.omega, h, capturing_planet.hash.value]
                    to_remove.append(h)
                    count_captured += 1

            if traj['collided'] and h not in to_remove:
                print(f'{h}: collided')
                to_remove.append(h)
                count_collided += 1

            if h in to_remove:
                sim.remove(hash=h)
                print(f'Particle {h} removed from simulation.')
                break

        print(f'At {sim.t} yr:', len(sim.particles)-sim.N_active, 'particles')
        

    end = time.time()

    print(f'core {core_id} done')
    with open(f'progress_tracking_files/progress_{tmax}_yr_{n_pl}_gas_drag_on_{pl_size}km.txt', 'a') as f:
        f.write(f'Core {core_id} finished and took {(end-start)/60:.3f} min.\n')

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

            c2 = np.all(dist[-1] < 87)

            if n_planets == 2:
                c3 = ((dist[0] < dist_small[1]))
            if n_planets == 3:
                # note that test particles are ordered from closest to furthest out!!
                c3 = ((dist[0] < dist_small[1])) and (dist[1] < dist_small[2]) 
                
            if c1 and c3 and c2:    
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
            else:
                print('file', file)
                                
    #n_particles = 200 * filecount
    #n_particles = len(mig_peri_list)
    print(f'Finally: {filecount} files')

    results = {
        'mig_all': mig_peri_list,
         #should be 8400 for this simulation 
    }

    #np.savez('hist_data_w_all_parameters.npz', **results)
    return results #{'migrated': migrated_array, 'ejected': ejected_array}

files = glob.glob(f'core_outputs_yr2/*50000_2_pl*saving_every_100_yrs*.nc')
results = find_mig_and_ej(files, 2)
mig_all = results['mig_all']

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

def prompt_gas_drag():
    while True:
        try:
            pl_size = int(input("What size planetesimals do you want to simulate? (1, 10, or 100 km): "))
            if pl_size in (1, 10, 100):
                return pl_size
            else:
                print("Please enter 1, 10, or 100.")
        except ValueError:
            print("no silly, I said 1, 10, or 100.")
    
def parallelization(tmax, N_cores, mig_all, n_pl, pl_size):

    with open(f'progress_tracking_files/progress_{tmax}_yr_{n_pl}_gas_drag_on_{pl_size}km.txt', 'w') as f:
        pass

    with open(f'progress_tracking_files/failed_cores_{n_pl}_gas_drag_{pl_size}km.txt', 'w') as f:
        pass


    print(f'You have {len(mig_all)} particles to choose from.')


    chunk_results = Parallel(n_jobs=N_cores, backend="loky", batch_size=1)(
        delayed(simulation)(tmax, ptcl, core_id, n_pl, pl_size)
        for core_id, ptcl in enumerate(mig_all)
    )
        
    return chunk_results


#%%
if __name__ == '__main__':

    tmax = 1e6
    N_cores = 56
    N_particles = N_cores

    two_pl_files = glob.glob(f'core_outputs_yr2/*50000_2_pl*saving*.nc')
    three_pl_files = glob.glob(f'core_outputs_yr2/*3_pl_*saving_every_100_yrs.nc')

    if prompt():                       # ← your existing prompt
        n_pl = prompt_n_planets()      # ← new prompt

        if n_pl == 2:
            files = two_pl_files
        elif n_pl == 3:
            files = three_pl_files

        mig_all = find_mig_and_ej(files, n_pl)['mig_all']
        print(len(mig_all))
        pl_size = prompt_gas_drag()
        filenames = parallelization(
           tmax, N_cores, mig_all, n_pl, pl_size
        )

    else:
        print('I think you should change your file name first. You are welcome.')