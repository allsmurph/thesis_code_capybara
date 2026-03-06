#%%
import numpy as np
import rebound
import astropy.constants as const
import astropy.units as u
import matplotlib.pyplot as plt
import time
import math

'''ignore this function! It uses everything from Eriksson et al. 2021. Later, I use pds 70 specific things'''
#def gas_drag(reb_sim):
    # print(type(sim_or_pointer))

    # if isinstance(sim_or_pointer, rebound.Simulation):
    #     sim = sim_or_pointer
    # else:
    #     sim = rebound.Simulation.from_pointer(sim_or_pointer)
    #sim = rebound.Simulation.from_pointer(sim_pointer)
    # '''defining some constants - DOUBLE CHECK WITH SUPERVISORS'''

    # M_star = sim.particles[0].m * u.M_sun
    # alpha = 1e-3 #as most papers
    # M_dot = 1e-10 *u.M_sun/u.yr #from Thanathibodee et al 2020.
    # r_out = 87 * u.AU
    # solid_rho = 1000 * u.kg / u.m**3
    # mu = 2.33
    # k_B = const.k_B

    # '''first, compute gas density as a function of z'''

    # for p in sim.particles[4:]:
    #     x, y, z = (p.x, p.y, p.z)*u.AU
    #     vx, vy, vz = (p.vx, p.vy, p.vz)*(u.AU/u.yr)
    #     r = p.a * u.AU

    #     Omega = np.sqrt(const.G * M_star / r**3)
    #     T = 150 * 0.35**(2/7) * 0.965**(-1/7) * (r/u.AU)**(-3/7) * u.K
        
    #     c_s = np.sqrt(k_B*T/(mu*const.m_p))
    #     H = c_s / Omega
    #     nu = alpha * Omega * H**2
    #     surface_density = (M_dot) / (3 * np.pi * nu) * np.exp(-r/r_out)
    
    #     gas_density = surface_density / (np.sqrt(2*np.pi) * H) * np.exp(-(z)**2 / (2 * H**2))

    #     'DEFINE GAS VELOCITY as v_K * unit vector e (what is e vec??)'
    #     v_K = np.sqrt(const.G * M_star / r).to('AU/yr')
    #     v_K_stripped = v_K.value
    #     v_gas = np.array([v_K_stripped*x/r, v_K_stripped*y/r, v_K_stripped*z/r]) * u.AU/u.yr
    
    #     v_rel = np.array([vx.value - v_gas[0].value, vy.value - v_gas[1].value, vz.value - v_gas[0].value])*u.AU/u.yr
    #     v_rel_mag = np.linalg.norm(v_rel)

    #     'calculate term in min func'
    #     lg = 5e-6 *u.kg / u.m**3 / gas_density
    #     R_pl = p.r * u.AU
    #     Re = 4 * R_pl * v_rel_mag / (c_s * lg)
    #     Re = Re.value
    #     C_D = 24/Re * (1+0.27*Re)**(0.43) + 0.47*(1-np.exp(-0.04*Re**(0.38)))
    #     v_th = np.sqrt(8/np.pi) * c_s
    #     cd_term = 3/8*v_rel_mag / v_th * C_D * Re
    #     '''next, compute the stopping time'''
    #     ts = (gas_density/solid_rho * v_th/R_pl * min(1, cd_term) )**(-1)

    #     a = -1/ts * v_rel
    #     a = a.value
    #     p.ax += a[0]
    #     p.ay += a[1]
    #     p.az += a[2]

#%%

'''converting constants to simulation units: AU, yr, M_sun '''

kgtomsun = 1/1.989e30 #kg to solar mass
kb = 3.0898292661510003e-61 #boltzmann constant in sim units
mtoau = 6.6846e-12 #meter to au
jtos = 0.000954588 #jupiter mass to solar mass
stoyr = 1/(365.25*24*3600)

'''initialising the simulation (no need for planets yet)'''
sim = rebound.Simulation()
sim.units = ('Msun', 'AU', 'yr')
sim.add(m=0.965)
# sim.add(m=0.5*jtos, a=10, e=0.25, f=np.random.rand()*2.*np.pi,)
# sim.add(m=1*jtos, a=21, e=0.11, f=np.random.rand()*2.*np.pi,)
# sim.add(m=3*jtos, a=34, e=0.03, f=np.random.rand()*2.*np.pi,)
sim.move_to_com() #centre of mass frame

'''add a/some planetesimal(s) of r=100 km (converted to au), somewhere in outer disc:'''

dists = np.linspace(70,80, 3)
for i in (dists):
    sim.add(m=0, a=i, f=np.random.rand()*2.*np.pi, e=0, r=6.68e-7)

sim.N_active = 1

#%

def gas_drag_2(reb_sim):
    '''
    in this new simulation, there's no longer need for Mdot, nu, or alpha
    (simplified gas surface density profile)
    '''

    '''defining some constants'''

    M_star = sim.particles[0].m 
    r_out = 40 
    solid_rho = 1000 * kgtomsun / mtoau**3
    mu = 2.33
    m_proton = 1.6726e-27 * kgtomsun
    G =  39.476926

    '''first, compute gas density as a function of z'''

    for p in sim.particles[1:]: #grab from simulation
        x, y, z = p.x, p.y, p.z
        vx, vy, vz = p.vx, p.vy, p.vz
        r = p.a

        Omega = np.sqrt(G * M_star / r**3)

        T = 38*(r/(40))**(-0.24)  #new temperature profile from pds 70 modelling papers
        
        c_s = np.sqrt(kb*T/(mu*m_proton)) #speed of sound 
        H = c_s / Omega
        
        sigma_0_in_correct_units = 3.03885e-7  #2.7 g/cm**2 in m_sun/AU**2 - also from pds 70 papers

        surface_density = sigma_0_in_correct_units * (r/r_out)**(-1) * np.exp(-r/r_out) #pds 70 papers
        #surface_density = 0.1
        
        if r>=18 and r<=40:
            surface_density *= 0.01 # add a gap

        gas_density = surface_density / (np.sqrt(2*np.pi) * H) * np.exp(-(z)**2 / (2 * H**2)) #from Eriksson+2021

        'DEFINE GAS VELOCITY'
        #almost keplerian - have to overcome gravity + pressure
        #(could multiply by correction value - in the slides of course) - skipped for now
        R = np.sqrt(x**2+y**2)
        v_K = np.sqrt(G * M_star / R)

        theta = np.arctan2(y,x)
        v_unit_vector = np.array([-np.sin(theta), np.cos(theta), 0]) #trig to get correct direction

        v_gas = v_K*v_unit_vector
        v_pl_vector = np.array([vx, vy, vz]) #velocity of planetesimal

        v_rel = v_pl_vector - v_gas 
        v_rel_mag = np.linalg.norm(v_rel)

        'calculate term in min func for stopping time - Eriksson+ 2021'
        lg = 5e-6 *kgtomsun / (mtoau**3 * gas_density) #convert units!
        R_pl = p.r 
        Re = 4 * R_pl * v_rel_mag / (c_s * lg)
        C_D = 24/Re * (1+0.27*Re)**(0.43) + 0.47*(1-np.exp(-0.04*Re**(0.38)))
        v_th = np.sqrt(8/np.pi) * c_s
        cd_term = 3/8*v_rel_mag / v_th * C_D * Re

        '''next, compute the stopping time'''
        ts = (gas_density/solid_rho * v_th/R_pl * min(1, cd_term) )**(-1)
        a = -1/ts * v_rel
        p.ax += a[0]
        p.ay += a[1]
        p.az += a[2]
        print(ts)

gas_drag_2(2)
#%%
'''Implementing the force with rebound - additional forces'''

sim.additional_forces = gas_drag_2

sim.force_is_velocity_dependent = 1

# '''Testing function in C'''
#%%
import rebound
import reboundx
print(reboundx.__file__)
import numpy as np
import astropy.constants as const
import astropy.units as u
import sys
import reboundx
import time
import matplotlib.pyplot as plt
import math

jtos = const.M_jup / const.M_sun

R_b = 2.72 * const.R_jup.to('au').value
R_c = 2.04 * const.R_jup.to('au').value
R_star = 1.26 * const.R_sun.to('au').value

a_b, a_c, a_d = 21.1, 35.3, 10.7
e_b, e_c, e_d = 0.131, 0.033, 0.25
w_b, w_c, w_d = 191.4, 63, 29
i_b, i_c, i_d = 128.7, 128.5, 151
Omega_b, Omega_c, Omega_d = 174.3, 159.8, 144

m_b, m_c, m_d = 1.5, 3, 0.5

dtor = np.pi / 180
# --- create normal REBOUND simulation ---
sim = rebound.Simulation()
sim.units = ('Msun','AU','yr')

sim.add(m = 0.965, x=0, y= 0, z = 0, hash='star')
sim.add(m = m_d*jtos, a = a_d, e = e_d, f=np.random.rand()*2.*np.pi,
        inc = (i_d-128.3)*dtor, omega = w_d*dtor, Omega =  Omega_d*dtor, hash='pd')
sim.add(m = m_b*jtos, a = a_b, e = e_b, f=np.random.rand()*2.*np.pi,
        omega = w_b*dtor, inc = (i_b-128.3)*dtor, Omega = Omega_b*dtor, hash = 'pb')
sim.add(m = m_c*jtos, a = a_c, e = e_c, f=np.random.rand()*2.*np.pi,
        omega = w_c*dtor, inc = (i_c-128.3)*dtor, Omega = Omega_c*dtor, hash = 'pc')

#%%
dists = np.linspace(80, 90, 2)

for i in dists:
    sim.add(m=0, a=i, f=np.random.rand()*2.*np.pi, e=0, r=6.68e-7)
# add planets and test particles ...
sim.N_active = 4

# --- attach REBOUNDx and load your plugin ---
rebx = reboundx.Extras(sim)

myforce = rebx.load_force("new_force")          # compiled above
rebx.add_force(myforce)

# --- integrate as usual ---
Nout = 10
times = np.linspace(0.,1000,1000)

N_particles = sim.N - sim.N_active

'''saving some stuff for plotting/diagnostics'''
semi_major = np.zeros((len(times), N_particles))
eccentricity = np.zeros((len(times), N_particles))
inclination = np.zeros((len(times), N_particles))

x_pos = np.zeros((len(times), N_particles))
y_pos = np.zeros((len(times), N_particles))
z_pos = np.zeros((len(times), N_particles))

x_pos_pl = np.zeros((len(times), sim.N_active-1))
y_pos_pl = np.zeros((len(times), sim.N_active-1))
z_pos_pl = np.zeros((len(times), sim.N_active-1))

tstart = time.time()
for i, t in enumerate(times):
    sim.integrate(t)
    # for pl in range(3):
    #     planet = sim.particles[pl+1]
    #     x_pos_pl[i, pl] = planet.x 
    #     y_pos_pl[i, pl] = planet.y
    #     z_pos_pl[i, pl] = planet.z

    for j in range(N_particles):
        p = sim.particles[sim.N_active + j]
        if math.isnan(np.any([p.ax, p.ay, p.az])):
            print('error')
            break
        semi_major[i, j] = p.a
        eccentricity[i, j] = p.e
        inclination[i, j] = p.inc
        x_pos[i, j] = p.x
        y_pos[i, j] = p.y
        z_pos[i, j] = p.z

tend = time.time()
print(f'{(tend - tstart)/60:.4f} min')


# %%
#%%
'''plotting a few different things'''
fig, axes = plt.subplots(3, 2, figsize=(14, 12))

# 1. Semi-major axis decayI
ax = axes[0, 0]
for j in range(N_particles):
    ax.plot(times, semi_major[:, j], label=f'Particle {j+1}', linewidth=2)
ax.set_xlabel('Time (years)', fontsize=12)
ax.set_ylabel('Semi-major axis (AU)', fontsize=12)
ax.set_title('Semimajor axis', fontsize=13, fontweight='bold')
ax.legend()
ax.grid(alpha=0.3)

# 2. Eccentricity damping
ax = axes[0, 1]
for j in range(N_particles):
    ax.plot(times, eccentricity[:, j], label=f'Particle {j+1}', linewidth=2)
ax.set_xlabel('Time (years)', fontsize=12)
ax.set_ylabel('Eccentricity', fontsize=12)
ax.set_title('Eccentricity', fontsize=13, fontweight='bold')
ax.legend()
ax.grid(alpha=0.3)

# 3. Inclination damping
ax = axes[1, 0]
for j in range(N_particles):
    ax.plot(times, np.degrees(inclination[:, j]), label=f'Particle {j+1}', linewidth=2)
ax.set_xlabel('Time (years)', fontsize=12)
ax.set_ylabel('Inclination (degrees)', fontsize=12)
ax.set_title('Inclination', fontsize=13, fontweight='bold')
ax.legend()
ax.grid(alpha=0.3)

# 5. xy trajectory
ax = axes[1, 1]
for j in range(N_particles):
    ax.plot(x_pos[:, j], y_pos[:, j], label=f'Particle {j+1}', alpha=0.7)
    # ax.scatter(x_pos[0, j], y_pos[0, j], s=100, marker='o', zorder=5)
    # ax.scatter(x_pos[-1, j], y_pos[-1, j], s=100, marker='x', zorder=5)
ax.scatter(0, 0, s=200, marker='*', color='gold', edgecolors='orange', 
          linewidths=2, label='Star', zorder=10)
ax.set_xlabel('x (AU)', fontsize=12)
ax.set_ylabel('y (AU)', fontsize=12)
ax.set_title('XY Trajectory', fontsize=13, fontweight='bold')
ax.axis('equal')
ax.legend()
ax.grid(alpha=0.3)


# 6. Vertical motion (z)
ax = axes[2, 0]
for j in range(N_particles):
    ax.plot(times, z_pos[:, j], label=f'Particle {j+1}', linewidth=2)
ax.set_xlabel('Time (years)', fontsize=12)
ax.set_ylabel('z (AU)', fontsize=12)
ax.set_title('Height (z) from plane', fontsize=13, fontweight='bold')
ax.legend()
ax.grid(alpha=0.3)

plt.tight_layout()
#plt.savefig('gas_drag_diagnostics.png', dpi=150, bbox_inches='tight')
plt.show()