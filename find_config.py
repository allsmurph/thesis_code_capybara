#%%

import numpy as np
import astropy.constants as const
import rebound
from matplotlib import pyplot as plt
from matplotlib.animation import FuncAnimation
from joblib import Parallel, delayed
import time

def simming(iteration):
        start = time.time()
        phases = np.round(np.random.rand(3)*2.*np.pi,3)

        sim = rebound.Simulation()
        sim.units = ['msun', 'yr', 'AU']

        sim.integrator = "ias15"  
        sim.dt = 0.1
        
        a_b, a_c, a_d = 21.1, 35.3, 10.7
        e_b, e_c, e_d = 0.131, 0.033, 0.25
        i_b, i_c, i_d = 128.7, 128.5, 151
        w_b, w_c, w_d = 191.4, 63, 29
        Omega_b, Omega_c, Omega_d = 174.3, 159.8, 144
        m_b, m_c, m_d = 1.5, 5, 0.5

        jtos = const.M_jup / const.M_sun

        sim.add(m = 0.965, hash='star')


        sim.add(m = m_d*jtos, a = a_d, e = e_d, omega=np.radians(w_d),
        f=phases[0],
        inc=np.radians(i_d-128.3), Omega = np.radians(Omega_d), hash='pd')

        sim.add(m=m_b*jtos, a = a_b, e = e_b, omega=np.radians(w_b),
        f=phases[1],
        inc=np.radians(i_b-128.3), Omega = np.radians(Omega_b), hash='pb')

        sim.add(m=m_c*jtos, a=a_c, e=e_c, omega=np.radians(w_c),
        f=phases[2],
        inc=np.radians(i_c-128.3), Omega=np.radians(Omega_c), hash='pc')   


        sim.move_to_com()

        tmax = 5000000
        times = np.linspace(0, tmax, int(tmax*5))
        
        crossed = False
        ended_above = False  # assume it stays above once crossed
        for t, tid in enumerate(times):
                if tid > times[0]:

                        dt = times[t] - times[t-1]
                        sim.integrate(sim.t + dt)

                        apo_d = sim.particles['pd'].a * (1 + sim.particles['pd'].e)
                        apo_b = sim.particles['pb'].a * (1 + sim.particles['pb'].e)
                        apo_c = sim.particles['pc'].a * (1 + sim.particles['pc'].e)

                        if np.all([apo_d, apo_b, apo_c]) > 0 and (apo_d < apo_b) and (apo_b < apo_c):
                                if not crossed:
                                        if apo_c >= 40:
                                                crossed = True
                                                print(f"apo_c reached 40.0 AU at t={sim.t:.2f}. Initial f={phases}")

                               # if crossed and t%50000 == 0:
   
                                if tid in [1000000, 2000000, 3000000, 4000000, 5000000] and crossed:
                                        if apo_c >= 40:
                                                ended_above = True

                                # elif tid == times[-1] and crossed:
                                #         if apo_c >= 40:
                                #                 ended_above = True
                                #                 #print(f"apo_c dropped below 40 again at t={sim.t:.2f}, apo_c={apo_c:.2f}")
                                #                 print(f'found a configuration. Saving phases data: {phases}')
                                #                 #stayed_above = False

                        else:
                                with open(f'discarded_sims.txt', 'a') as file:
                                        file.write(f'Discarded sim {iteration} with phases {phases}. Time: {tid} yr \n')
                                
                                print('something bad happened. abort!')
                                break

        end = time.time()
        if crossed and ended_above:
                print(f'iteration {iteration} finished and took {(end-start)/60:.3f} min.')
                with open(f'potential_phases.txt', 'a') as f:
                        f.write(f'Found a phase combo that works: {phases} \n')
                        
                # print(sim.orbits())


        # for p, particle in enumerate(sim.particles[1:]):
        #         a = particle.a * (1+particle.e)
        #         print(f'{p}: {a} au; phases: {phases}')

        return None


# for i in range(30):
#         print(simming(i))

n=50
results = Parallel(n_jobs=n)(delayed(simming)(i) for i in range(n))

#print(results)

# with open(f'potential_phases.txt', 'w') as f:
#         pass