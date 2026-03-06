#%%

import numpy as np
import astropy.constants as const
import rebound
from matplotlib import pyplot as plt
from matplotlib.animation import FuncAnimation
import time
#%%
'''Set up the simulation with known values'''

def check_sim(phase_d, phase_b, phase_c):
        sim = rebound.Simulation()
        sim.units = ['msun', 'yr', 'AU']

        sim.integrator = "trace"  

        a_b, a_c, a_d = 21.1, 35.3, 10.7
        e_b, e_c, e_d = 0.131, 0.033, 0.25
        i_b, i_c, i_d = 128.7, 128.5, 151
        w_b, w_c, w_d = 191.4, 63, 29
        Omega_b, Omega_c, Omega_d = 174.3, 159.8, 144
        m_b, m_c, m_d = 1.5, 5, 0.5

        #phase_d, phase_b, phase_c =
        
        #2.377, 1.423, 5.951

        #3.8, 3.99, 0.169 -- works!!

        #4.45, 1.786, 4.441 -- pds 70 c goes to 137 au

        #2.496, 0.838, 4.701 -- this one breaks down eventually, high eccentricites, but could work


        jtos = const.M_jup / const.M_sun

        sim.add(m = 0.965, hash='star')

        sim.add(m = m_d*jtos, a = a_d, e = e_d, omega=np.radians(w_d),
                f=phase_d,
                inc=np.radians(i_d-128.3), Omega = np.radians(Omega_d), hash='pd')

        sim.add(m=m_b*jtos, a = a_b, e = e_b, omega=np.radians(w_b),
                f=phase_b,
                inc=np.radians(i_b-128.3), Omega = np.radians(Omega_b), hash='pb')

        sim.add(m=m_c*jtos, a=a_c, e=e_c, omega=np.radians(w_c),
                f=phase_c,
                inc=np.radians(i_c-128.3), Omega=np.radians(Omega_c), hash='pc')   


        sim.move_to_com()
        # print(sim.orbits())
        r_target = 40.0  # AU
        star = sim.particles['star']
        pc = sim.particles['pc']


        tmax = 5.5e6
        times = np.linspace(0, tmax, int(tmax*5))
        Nsample = 5000     # how many output points we want
        sample_indices = np.linspace(0, len(times)-1, Nsample, dtype=int)
        t_save = np.zeros((Nsample, 1))
        apos = np.zeros((Nsample, 3))
        pos = np.zeros((Nsample, 3)) # time, planet, x/y/z

        counter = 0 

        for t, tid in enumerate(times):
                if tid > times[0]:
                        dt = times[t] - times[t-1]
                        sim.integrate(sim.t + dt)


                if t in sample_indices:
                        
                        orb_d, orb_b, orb_c = sim.particles['pd'], sim.particles['pb'], sim.particles['pc']
                        
                        r_ap_d = orb_d.a * (1 + orb_d.e)
                        r_ap_b = orb_b.a * (1 + orb_b.e)
                        r_ap_c = orb_c.a * (1 + orb_c.e)
                        rs = [r_ap_d, r_ap_b, r_ap_c]

                        positions = [np.sqrt(orb_d.x**2 + orb_d.y**2 + orb_d.z**2), np.sqrt(orb_b.x**2 + orb_b.y**2 + orb_b.z**2), np.sqrt(orb_c.x**2 + orb_c.y**2 + orb_c.z**2)]

                        t_save[counter] = sim.t
                        for i in range(3):
                                apos[counter][i] = rs[i]
                                pos[counter][i] = positions[i]
                        counter += 1
                        
                        # if r_ap >= r_target:
                        #         print(f"Planet c reached apoastron {r_ap:.3f} AU at time = {t:.2f}")
                        #         print(orb_c.a)
                        
                                #break
        # sim.integrate(5e6)
        # print(sim.orbits())


        fig, ax = plt.subplots(1, figsize=(12, 8))
        plt.plot(t_save, pos, label=[f'PDS 70 d, f= {phase_d}', f'PDS 70 b, f={phase_b}', f'PDS 70 c, f= {phase_c}'])
        plt.axhline(y=40, color='black', linestyle='--')
        plt.title('XY pos over time', fontsize=16)
        plt.xlabel('t [Myr]')
        plt.ylabel(r'$R_a$ [AU]')
        plt.legend(fontsize=12, loc='lower left')
        plt.savefig(f'plots/positions_{phase_c}_long_time.png')

        return fig

res1 = check_sim(6.274, 2.758, 5.312)

print(res1)
#t_save
#%%
# for p, particle in enumerate(sim.particles[1:]):
#     a = particle.a * (1+particle.e)
#     print(a)

#sim.save_to_file('postsim.bin')

# fig = plt.figure(figsize=(8,8))
# ax = plt.subplot(111)
# ax.set_xlabel('x [au]')
# ax.set_ylabel('y [au]')
# ax.set_aspect('equal')

# def update(i):
#     ax.clear()
#     sim.integrate(5e6 + i*5)
#     op = rebound.OrbitPlot(sim, fig=fig, ax = ax, xlim = (-80, 80), ylim= (-80, 80), color=True, unitlabel='[au]')
#     op.primary.set_color("orange")
#     op.particles.set_color(["red", "turquoise", 'magenta'])
#     ax.set_title(f'The PDS 70 system at t = {sim.t} yr ', size=14)
#     ax.text(x = -70, y = 70, s='PDS 70 d', color='red', size=12)
#     ax.text(x = -70, y = 66, s='PDS 70 b', color='turquoise', size=12)
#     ax.text(x = -70, y = 62, s='PDS 70 c', color='magenta', size=12)
    
#     return op


# ani = FuncAnimation(fig, update, frames=200, interval=50, blit=False)

# ani.save('test_animation_3_pl.gif', writer='pillow', fps = 10)

# for p, particle in enumerate(sim.particles[1:]):
#     a = particle.a * (1+particle.e)
#     print(a)

# #%%
# sim.orbits()