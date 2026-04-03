#%%
import numpy as np
import rebound
import astropy.constants as const
import astropy.units as u
import matplotlib.pyplot as plt
import time
import math
from plotting_params import use_my_style
use_my_style()

R_char = 40 
jtos = 0.000954588
r_pl = 6.684587122268445e-07
solid_rho = 1683721.7643842339
mu = 2.33
m_proton = 8.411856872862986e-58 
sigma_char = 3.038845902395208e-07
lg = 0.00841860882192117 
kb = 3.0898292661510003e-61
stellar_radius = const.R_sun.to('AU').value
kgtomsun = (1 / const.M_sun).value
dtor = np.pi / 180
# --- create normal REBOUND simulation ---
sim = rebound.Simulation()
sim.units = ('Msun','AU','yr')

sim.add(m = 0.965, x=0, y= 0, z = 0, hash='star')


for e in np.linspace(0.05, 0.99, 100):
    for a in np.linspace(0.3, 18, 50):
        sim.add(m=0, a=a, e=e, f=np.pi/2, r=6.68e-7)

# dists = np.linspace(5, 10, 2)
# for i in dists:
#     sim.add(m=0, a=i, f=np.random.rand()*2.*np.pi, e=0, r=6.68e-7)
# add planets and test particles ...
sim.N_active = 1
sim.integrate(0.001)

def get_GD_terms(reb_sim):
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

    gas_densities = []
    C_Ds = []
    Ts = []
    v_rel_mags = []

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
        if r<stellar_radius*3.4:
            gas_density = 0 #no gas inside of 3.4 stellar radii - from pds 70 papers

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
        #v_th = np.sqrt(8/np.pi) * c_s
        #cd_term = 3/8*v_rel_mag / v_th * C_D * Re

        '''next, compute the stopping time'''
        #ts = (gas_density/solid_rho * v_th/r_pl * min(1, cd_term) )**(-1)
        # a = -1/ts * v_rel
        # p.ax += a[0]
        # p.ay += a[1]
        # p.az += a[2]
        gas_densities.append(gas_density)
        C_Ds.append(C_D)
        Ts.append(T)
        v_rel_mags.append(v_rel_mag)

    return gas_densities, C_Ds, Ts, v_rel_mags


'''remember units are AU, yr, Msun'''

def P_sat_vap(T_pl):
    Tt = 273.16 #Kelvin
    Pt = ((0.0061e-3 * u.bar).to('Msun/(AU*yr2)')).value

    coeffs = [20.9969665197897, 3.72437478271362, -13.9205483215524,
                29.6988765013566, -40.1972392635944, 29.7880481050215, -9.13050963547721]
    
    poli = sum(coeffs[i] * (T_pl/Tt)**i for i in range(len(coeffs)))
    t1 = 3/2 * np.log(T_pl/Tt) + (1 - Tt/T_pl) * poli

    ans = np.exp(t1) * Pt
    return ans

def P_sat_vap_IAPWS(T_pl):
    Pc = ((22.065e6 * u.Pa).to('Msun/(AU*yr2)')).value
    Tc = 647.096 #Kelvin
    theta = T_pl / Tc
    tau = 1 - theta
    a1, a2, a3, a4, a5, a6 = -7.85951783, 1.84408259, -11.7866497, 22.6807411, -15.9618719, 1.80122502
    log_p_over_pc = Tc/T_pl * (a1*tau + a2*tau**1.5 + a3*tau**3 + a4*tau**3.5 + a5*tau**4 + a6*tau**7.5)
    Psat = np.exp(log_p_over_pc) * Pc

    return Psat

Ts = np.linspace(1, 600, 1000)
plt.plot(Ts, [(P_sat_vap_IAPWS(T)*u.Msun/(u.AU*u.yr**2)).to('Pa').value for T in Ts], label='IAPWS')
plt.plot(Ts, [(P_sat_vap(T)*u.Msun/(u.AU*u.yr**2)).to('Pa').value for T in Ts], label='poly fit')
plt.yscale('log')
plt.ylim(1e-52, 1e52)
plt.xlabel('Planetesimal temperature [K]')
plt.ylabel('Saturation vapor pressure [Pa]')
plt.grid()
plt.legend()
plt.show()

#%%)
def func(T_pl):

    L_w = ((2.8e6 * u.J / u.kg).to('AU2/yr2')).value
    sigma_sb = (const.sigma_sb.to('Msun/(yr3*K4)')).value
    mu = ((18 * u.g/u.mol).to('Msun/mol')).value
    R_g = ((8.314 * u.J / (u.mol * u.K)).to('Msun*AU2/(yr2*K*mol)')).value


    plt.show()

    gas_density, C_D, T, v_rel_mag = get_GD_terms(sim)

    term2 = C_D * gas_density * v_rel_mag**3 / (32 * sigma_sb)
    term3 = P_sat_vap(T_pl) / (sigma_sb) * np.sqrt(mu / (8*np.pi*R_g*T_pl)) * L_w
    
    return #T**4 + term2 - term3 - T_pl**4

temps = np.linspace(1, 200, 100)
plt.plot(temps, func(temps))
plt.yscale('log')
plt.ylim(1e-50, 1e10)
plt.xlabel('Planetesimal temperature [K]')
plt.ylabel('f(T_pl)')
plt.grid()
plt.show()
#%%
def bisection(a,b):

    if (func(a) * func(b) >= 0):
        print("You have not assumed right a and b\n")
        return
    
    c = a
    
    while (np.abs(b-a) >= 1):

        # Find middle point
        c = (a+b)/2
 
        # Check if middle point is root
        if (func(c) == 0.0):
            break
 
        # Decide the side to repeat the steps
        if (func(c)*func(a) < 0):
            b = c
        else:
            a = c     
    print("The value of root is : ","%.4f"%c)
    

bisection(0, 1000)

#%%
'''At what periastron does second term of temp expression explode?'''

gds, C_Ds, Ts, v_rel_mags = get_GD_terms(sim)
sigma_sb = (const.sigma_sb.to('Msun/(yr3*K4)')).value

term2 = (np.array(C_Ds) * np.array(gds) * (np.array(v_rel_mags))**3 / (32 * sigma_sb))**(1/4)

all_es = [sim.particles[i+1].e for i in range(len(sim.particles)-sim.N_active)]
all_as = [sim.particles[i+1].a for i in range(len(sim.particles)-sim.N_active)]

plt.plot(np.array(all_as) * (1 - np.array(all_es)), term2, c='saddlebrown', linestyle='', marker='o', markersize=1)
plt.xlabel('Periastron [AU]')
plt.ylabel('Friction heating term [K]')
plt.yscale('log')
plt.xscale('log')
plt.grid(alpha=0.5)
plt.show()

plt.plot(np.array(all_es), term2, c='orange', linestyle='',marker='o', markersize=1)
plt.xlabel('Eccentricity')
plt.ylabel('Friction heating term [K]')
plt.yscale('log')
plt.xscale('log')
plt.grid(alpha=0.5)
plt.show()

#%%
'''Solve equation by finding where LHS and RHS intersect'''

# sim2 = rebound.Simulation()

# sim2.units = ('Msun','AU','yr')

# sim2.add(m = 0.965, x=0, y= 0, z = 0, hash='star')


# sim2.add(m=0, a=0.5, e=0.5, f=np.pi/2, r=6.68e-7)

# sim2.N_active = 1
# sim2.integrate(0.001)

def T_pl_dependent_terms(T_pl):
    L_w = ((2.8e6 * u.J / u.kg).to('AU2/yr2')).value
    sigma_sb = (const.sigma_sb.to('Msun/(yr3*K4)')).value
    mu = ((18 * u.g/u.mol).to('Msun/mol')).value
    R_g = ((8.314 * u.J / (u.mol * u.K)).to('Msun*AU2/(yr2*K*mol)')).value

    term3 = P_sat_vap_IAPWS(T_pl) / (sigma_sb) * np.sqrt(mu / (8*np.pi*R_g*T_pl)) * L_w

    return term3 + T_pl**4
    
def T_pl_independent_terms(simulation):
    sigma_sb = (const.sigma_sb.to('Msun/(yr3*K4)')).value

    gds, C_Ds, T, v_rel_mags = get_GD_terms(simulation)

    term2 = (np.array(C_Ds) * np.array(gds) * (np.array(v_rel_mags))**3 / (32 * sigma_sb))**(1/4)

    return term2 + np.array(T)**4

Ts = np.linspace(1, 600, 1000)

f_T = T_pl_dependent_terms(Ts)[:,None]
min_t_ind, max_t_ind = np.min(T_pl_independent_terms(sim)), np.max(T_pl_independent_terms(sim))

diff_min = (f_T - min_t_ind).flatten()
diff_max = (f_T - max_t_ind).flatten()

# sign changes → intersections
idx_min = np.where(np.diff(np.sign(diff_min).flatten()) != 0)[0]
idx_max = np.where(np.diff(np.sign(diff_max)) != 0)[0]

# extract temperatures
T_int_min = Ts[idx_min]
T_int_max = Ts[idx_max]


plt.plot(Ts, T_pl_dependent_terms(Ts)[:,None], label=r'subl cooling + $T_\mathrm{pl}^4$', c='saddlebrown')
# plt.plot(Ts, np.full_like(Ts, T_pl_independent_terms(sim)[min_t_ind]), c='orange', alpha=0.5)
# plt.plot(Ts, np.full_like(Ts, T_pl_independent_terms(sim)[max_t_ind]), c='orange', alpha=0.5)
#plt.scatter(T_int_min, min_t_ind, color='black', label='T_pl independent terms (min)', zorder=5)
#plt.scatter(T_int_max, max_t_ind, color='black', label='T_pl independent terms (max)', zorder=5)
plt.fill_between(Ts, min_t_ind, max_t_ind, color='orange', alpha=0.5, label='friction + rad heating')
# plt.vlines(T_int_min[0], ymin=1, ymax=min_t_ind, linestyle='--', colors='grey')
# plt.vlines(T_int_max[0], ymin=1, ymax=max_t_ind, linestyle='--', colors='grey')
#plt.plot(Ts, Ts**4, label=r'$T_\mathrm{pl}^4$', c='steelblue', linestyle='--')
# for i in range(400):
#     plt.plot(Ts, np.full_like(Ts, T_pl_independent_terms(sim)[-i]), c='orange', alpha=0.5)
#     #plt.plot(Ts, np.full_like(Ts, T_pl_independent_terms(sim)[-i]), c='orange')
#     if i == 0:
#         plt.plot(Ts, np.full_like(Ts, T_pl_independent_terms(sim)[i]), label=f'T_pl independent terms', c='orange')
plt.ylim(1e0, 1e16)
plt.yscale('log')
plt.xlabel('Planetesimal temperature [K]')
plt.ylabel('arbitrary units')
plt.grid()
plt.legend(loc='lower right')
plt.title(r'$T_\mathrm{pl}^4$ + subl cooling = frict + rad heating', y=1.02, fontsize=16)
plt.tight_layout()
plt.show()

