#%%
import numpy as np
import rebound
import astropy.constants as const
import astropy.units as u
import matplotlib.pyplot as plt
import rebound
import time
import math

R_char = 40 
jtos = 0.000954588
r_pl = 6.684587122268445e-07
solid_rho = 1683721.7643842339
mu = 2.33
m_proton = 8.411856872862986e-58 
sigma_char = 3.038845902395208e-07
lg = 0.00841860882192117 
kb = 3.0898292661510003e-61

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

sim.add(m = 0.965, x=0, y= 0, z = 0, hash='star', r = R_star)

# sim.add(m = m_d*jtos, a = a_d, e = e_d, f=np.random.rand()*2.*np.pi,
#         inc = (i_d-128.3)*dtor, omega = w_d*dtor, Omega =  Omega_d*dtor, hash='pd')
# sim.add(m = m_b*jtos, a = a_b, e = e_b, f=np.random.rand()*2.*np.pi,
#         omega = w_b*dtor, inc = (i_b-128.3)*dtor, Omega = Omega_b*dtor, hash = 'pb', r=R_b)
# sim.add(m = m_c*jtos, a = a_c, e = e_c, f=np.random.rand()*2.*np.pi,
#         omega = w_c*dtor, inc = (i_c-128.3)*dtor, Omega = Omega_c*dtor, hash = 'pc', r=R_c)

# sim.add(m = m_d*jtos, a = a_d)
# sim.add(m = m_b*jtos, a = a_b)
# sim.add(m = m_c*jtos, a = a_c)


dists = np.linspace(5, 10, 2)
for i in dists:
    sim.add(m=0, a=i, f=np.random.rand()*2.*np.pi, e=0, r=6.68e-7)
# add planets and test particles ...
sim.N_active = 1
sim.integrate(10)
#%%
def gas_drag(reb_sim):

    G = sim.G
    M_star = sim.particles[0].m 

    for p in sim.particles[sim.N_active:]:
        x, y, z = p.x, p.y, p.z
        vx, vy, vz = p.vx, p.vy, p.vz
        R = p.a 
        Omega = np.sqrt(G * M_star / R**3)

        T = 44 * (R / 22)**(-0.24) #from an Ideal testbed ... Bae et al. 2019 

        cs = np.sqrt(kb*T/(mu*m_proton))
        H = cs/Omega #in AU

        gas_surface_density = sigma_char * (R/R_char)**(-1) * np.exp(-R/R_char)

        if R>=18 and R<=40:
            gas_surface_density *= 0.01
        
        gas_density = gas_surface_density / (np.sqrt(2*np.pi)* H) * np.exp(-z**2/(2*H**2))

        v_K = np.sqrt(G * M_star / R)
        theta = np.arctan2(y,x)
        v_unit_vector = np.array([-np.sin(theta), np.cos(theta), 0])

        v_gas = v_K * v_unit_vector
        v_pl_vector = np.array([vx, vy, vz])
        v_rel = v_gas - v_pl_vector
        v_rel_mag = np.linalg.norm(v_rel)

        Re = 4 * r_pl * v_rel_mag / (cs * lg)

        C_D = 24/Re * (1+0.27*Re)**(0.43)+ 0.47 * (1-np.exp(-0.04*Re**0.38))

        v_th = np.sqrt(8/np.pi) * cs

        C_D_term = 3/8* v_rel_mag/v_th*C_D 

        ts = (gas_density / solid_rho * v_th / r_pl * min(1, C_D_term))**(-1)


        a = -1/ts * v_rel

        p.ax += a[0]
        p.ay += a[1]
        p.az += a[2]

        print('vrel', v_rel)


gas_drag(sim)
#%%
kgtomsun = 1/1.989e30 #kg to solar mass
mtoau = mtoau = 6.6846e-12 #meter to au

def gas_drag_2(reb_sim):
    '''
    in this new simulation, there's no longer need for Mdot, nu, or alpha
    (simplified gas surface density profile)
    '''

    '''defining some constants'''

    r_out = 40 
    mu = 2.33
    m_proton = 1.6726e-27 * kgtomsun

    G = sim.G
    M_star = sim.particles[0].m 

    '''first, compute gas density as a function of z'''

    for p in sim.particles[sim.N_active:]: #grab from simulation
        x, y, z = p.x, p.y, p.z
        vx, vy, vz = p.vx, p.vy, p.vz
        r = np.sqrt(x**2+y**2)

        Omega = np.sqrt(G * M_star / r**3)

        T = 38*(r/(40))**(-0.24)  #new temperature profile from pds 70 modelling papers
        
        c_s = np.sqrt(kb*T/(mu*m_proton)) #speed of sound 
        H = c_s / Omega
        
        #sigma_0_in_correct_units = 3.03885e-7  #2.7 g/cm**2 in m_sun/AU**2 - also from pds 70 papers

        surface_density = sigma_char * (r/r_out)**(-1) * np.exp(-r/r_out) #pds 70 papers
        #surface_density = 0.1
        
        if r>=18 and r<=40:
            surface_density *= 0.01 # add a gap

        gas_density = surface_density / (np.sqrt(2*np.pi) * H) * np.exp(-(z)**2 / (2 * H**2)) #from Eriksson+2021

        'DEFINE GAS VELOCITY'
        #almost keplerian - have to overcome gravity + pressure
        #(could multiply by correction value - in the slides of course) - skipped for now
        v_K = np.sqrt(G * M_star / r)

        theta = np.arctan2(y,x)
        v_unit_vector = np.array([-np.sin(theta), np.cos(theta), 0]) #trig to get correct direction

        v_gas = v_K*v_unit_vector
        v_pl_vector = np.array([vx, vy, vz]) #velocity of planetesimal

        v_rel = v_pl_vector - v_gas 
        v_rel_mag = np.linalg.norm(v_rel)

        'calculate term in min func for stopping time - Eriksson+ 2021'
        #g = 5e-6 *kgtomsun / (mtoau**3 * gas_density) #convert units!
        #R_pl = p.r 
        Re = 4 * r_pl * v_rel_mag / (c_s * lg)
        C_D = 24/Re * (1+0.27*Re)**(0.43) + 0.47*(1-np.exp(-0.04*Re**(0.38)))
        v_th = np.sqrt(8/np.pi) * c_s
        cd_term = 3/8*v_rel_mag / v_th * C_D * Re

        '''next, compute the stopping time'''
        ts = (gas_density/solid_rho * v_th/r_pl * min(1, cd_term) )**(-1)
        a = -1/ts * v_rel
        p.ax += a[0]
        p.ay += a[1]
        p.az += a[2]

#%%

sim.additional_forces = gas_drag


sim.force_is_velocity_dependent = 1


#%%

Nout = 10
times = np.linspace(0.,10000,10000)

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
inc_pl =  np.zeros((len(times), sim.N_active-1))
a_pl =  np.zeros((len(times), sim.N_active-1))

tstart = time.time()
for i, t in enumerate(times):
    sim.integrate(t)
    for pl in range(sim.N_active-1):
        planet = sim.particles[pl+1]
        x_pos_pl[i, pl] = planet.x 
        y_pos_pl[i, pl] = planet.y
        z_pos_pl[i, pl] = planet.z
        inc_pl[i, pl] = planet.inc
        a_pl[i, pl] = planet.a

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

#%%
# 
print(np.degrees(inc_pl))
colors = ["#FA5812", "#F77852", '#FF5E5B']
c2 = ["#0F6A94", '#124559', "#91DCF3" ]
'''plotting a few different things'''
fig, axes = plt.subplots(3, 2, figsize=(14, 12))

# 1. Semi-major axis decayI
ax = axes[0, 0]
for j in range(N_particles):
    ax.plot(times, semi_major[:, j], color=colors[j], label=f'Particle {j+1}', linewidth=1)
for i in range(sim.N_active-1):
    ax.plot(times,a_pl[:, i], color=c2[i], label=f'Planet {i+1}', linewidth=1)

ax.set_xlabel('Time (years)', fontsize=12)
ax.set_ylabel('Semi-major axis (AU)', fontsize=12)
ax.set_title('Semimajor axis', fontsize=13, fontweight='bold')
ax.legend()
ax.grid(alpha=0.3)

# 2. Eccentricity damping
ax = axes[0, 1]
for j in range(N_particles):
    ax.plot(times, eccentricity[:, j], color=colors[j], label=f'Particle {j+1}', linewidth=2)
ax.set_xlabel('Time (years)', fontsize=12)
ax.set_ylabel('Eccentricity', fontsize=12)
ax.set_title('Eccentricity', fontsize=13, fontweight='bold')
ax.legend()
ax.grid(alpha=0.3)

# 3. Inclination damping
ax = axes[1, 0]
for j in range(N_particles):
   ax.plot(times, np.degrees(inclination[:, j]), color=colors[j], label=f'Particle {j+1}', linewidth=2)
for i in range(sim.N_active-1):
    ax.plot(times, np.degrees(inc_pl[:, i]), color=c2[i], label=f'Planet {i+1}', linewidth=2)

ax.set_xlabel('Time (years)', fontsize=12)
ax.set_ylabel('Inclination (degrees)', fontsize=12)
ax.set_title('Inclination', fontsize=13, fontweight='bold')
ax.legend()
ax.grid(alpha=0.3)

# 5. xy trajectory
ax = axes[1, 1]
for j in range(N_particles):
    ax.plot(x_pos[:, j], y_pos[:, j], color=colors[j], label=f'Particle {j+1}', alpha=0.7)
    # ax.scatter(x_pos[0, j], y_pos[0, j], s=100, marker='o', zorder=5)
    # ax.scatter(x_pos[-1, j], y_pos[-1, j], s=100, marker='x', zorder=5)
ax.scatter(0, 0, s=200, marker='*', color='gold', edgecolors='orange', 
          linewidths=2, label='Star', zorder=10)
ax.set_xlabel('x (AU)', fontsize=12)
ax.set_ylabel('y (AU)', fontsize=12)
ax.set_title('XY Trajectory Planetesimals', fontsize=13, fontweight='bold')
ax.axis('equal')
ax.legend()
ax.grid(alpha=0.3)


# 6. Vertical motion (z)
ax = axes[2, 0]
for j in range(N_particles):
    ax.plot(times, z_pos[:, j], color=colors[j], label=f'Particle {j+1}', linewidth=2)
for i in range(sim.N_active -1):
    ax.plot(times, z_pos_pl[:, j], color=c2[i], label=f'Planet {i+1}', linewidth=2)

ax.set_xlabel('Time (years)', fontsize=12)
ax.set_ylabel('z (AU)', fontsize=12)
ax.set_title('Height (z) from plane', fontsize=13, fontweight='bold')
ax.legend()
ax.grid(alpha=0.3)


ax = axes[2,1]
for j in range(3):
    ax.plot(x_pos_pl[:, j], y_pos_pl[:, j], color=c2[j], label=f'Planet {j+1}', alpha=0.7)
    # ax.scatter(x_pos[0, j], y_pos[0, j], s=100, marker='o', zorder=5)
    # ax.scatter(x_pos[-1, j], y_pos[-1, j], s=100, marker='x', zorder=5)
# ax.scatter(0, 0, s=200, marker='*', color='gold', edgecolors='orange', 

#           linewidths=2, label='Star', zorder=10)
ax.set_xlabel('x (AU)', fontsize=12)
ax.set_ylabel('y (AU)', fontsize=12)
ax.set_title('XY Trajectory Planets', fontsize=13, fontweight='bold')
ax.axis('equal')
ax.legend()
ax.grid(alpha=0.3)

plt.suptitle('PDS 70 planets, python gas drag 1', fontsize=16, fontweight='bold')
plt.tight_layout()
#plt.savefig('gas_drag_diagnostics.png', dpi=150, bbox_inches='tight')

plt.show()