/** * @file central_force.c
 * @brief   A general central force.
 * @author  Dan Tamayo <tamayo.daniel@gmail.com>
 * 
 * @section     LICENSE
 * Copyright (c) 2015 Dan Tamayo, Hanno Rein
 *
 * This file is part of reboundx.
 *
 * reboundx is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * reboundx is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with rebound.  If not, see <http://www.gnu.org/licenses/>.
 *
 * The section after the dollar signs gets built into the documentation by a script.  All lines must start with space * space like below.
 * Tables always must be preceded and followed by a blank line.  See http://docutils.sourceforge.net/docs/user/rst/quickstart.html for a primer on rst.
 * $$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
 *
 * $Central Force$       // Effect category (must be the first non-blank line after dollar signs and between dollar signs to be detected by script).
 *
 * ======================= ===============================================
 * Authors                 D. Tamayo
 * Implementation Paper    `Tamayo, Rein, Shi and Hernandez, 2019 <https://ui.adsabs.harvard.edu/abs/2020MNRAS.491.2885T/abstract>`_.
 * Based on                None
 * C Example               :ref:`c_example_central_force`
 * Python Example          `CentralForce.ipynb <https://github.com/dtamayo/reboundx/blob/master/ipython_examples/CentralForce.ipynb>`_.
 * ======================= ===============================================
 * 
 * Adds a general central acceleration of the form a=Acentral*r^gammacentral, outward along the direction from a central particle to the body.
 * Effect is turned on by adding Acentral and gammacentral parameters to a particle, which will act as the central body for the effect,
 * and will act on all other particles.
 *
 * **Effect Parameters**
 * 
 * None
 *
 * **Particle Parameters**
 *
 * ============================ =========== ==================================================================
 * Field (C type)               Required    Description
 * ============================ =========== ==================================================================
 * Acentral (double)             Yes         Normalization for central acceleration.
 * gammacentral (double)         Yes         Power index for central acceleration.
 * ============================ =========== ==================================================================
 * 
 */

// #include <stdio.h>
// #include <stdlib.h>
// #include <math.h>
// #include "rebound.h"
// #include "reboundx.h"

// // Physical constants (in simulation units: Msun, AU, yr)
// static const double R_CHAR = 40.0;
// static const double JTOS = 0.000954588;
// static const double R_PL = 6.684587122268445e-07;
// static const double SOLID_RHO = 1683721.7643842339;
// static const double MU = 2.33;
// static const double M_PROTON = 8.411856872862986e-58;
// static const double SIGMA_CHAR = 3.038845902395208e-07;
// static const double LG = 0.00841860882192117;
// static const double KB = 3.0898292661510003e-61;

// void test_force(struct reb_simulation* const sim, struct rebx_force* const force, struct reb_particle* const particles, const int N) {
    
//     // Safety checks
//     if (sim == NULL || particles == NULL || N < 2) return;
    
//     const double G = sim->G;
//     const double M_star = particles[0].m;
//     const int N_active = sim->N_active;
    
//     // Loop over test particles only
//     for (int i = N_active; i < N; i++) {
//         struct reb_particle* p = &particles[i];
        
//         const double x = p->x;
//         const double y = p->y;
//         const double z = p->z;
//         const double vx = p->vx;
//         const double vy = p->vy;
//         const double vz = p->vz;  // BUG: This should be p->vz!
     
//         // Calculate orbital elements to get semimajor axis
//         struct reb_orbit orb = reb_orbit_from_particle(G, *p, particles[0]);
        
//         // Safety check for valid orbit
//         if (orb.a <= 0.0 || !isfinite(orb.a)) continue;
        
//         const double R = sqrt(x*x + y*y);
//         const double a = orb.a;
//         const double Omega = sqrt(G * M_star / (a*a*a));
        
//         // Safety check for Omega
//         if (!isfinite(Omega) || Omega <= 0.0) continue;
        
//         // Temperature profile (Bae et al. 2019)
//         const double T = 44.0 * pow(R / 22.0, -0.24);
        
//         // Sound speed and scale height
//         const double cs = sqrt(KB * T / (MU * M_PROTON));
//         if (!isfinite(cs) || cs <= 0.0) continue;
        
//         const double H = cs / Omega;
//         if (!isfinite(H) || H <= 0.0) continue;
        
//         // Surface density profile
//         double gas_surface_density = SIGMA_CHAR * pow(R / R_CHAR, -1.0) * exp(-R / R_CHAR);
        
//         // Gap between R=18 and R=40
//         if (R >= 18.0 && R <= 40.0) {
//             gas_surface_density *= 0.01;
//         }
        
//         // Vertical density profile (Gaussian)
//         const double gas_density = gas_surface_density / (sqrt(2.0 * M_PI) * H) * 
//                                    exp(-z*z / (2.0 * H*H));
        
//         // Keplerian velocity at radius R
//         const double v_K = sqrt(G * M_star / R);
//         const double theta = atan2(y, x);
        
//         // Gas velocity (azimuthal Keplerian flow)
//         const double v_gas_x = -v_K * sin(theta);
//         const double v_gas_y = v_K * cos(theta);
//         const double v_gas_z = 0.0;
        
//         // Relative velocity
//         const double v_rel_x = v_gas_x - vx;
//         const double v_rel_y = v_gas_y - vy;
//         const double v_rel_z = v_gas_z - vz;
//         const double v_rel_mag = sqrt(v_rel_x*v_rel_x + v_rel_y*v_rel_y + v_rel_z*v_rel_z);
        
//         // Avoid division by zero
//         if (v_rel_mag <= 0.0) continue;
        
//         // Reynolds number
//         const double Re = 4.0 * R_PL * v_rel_mag / (cs * LG);
        
//         // Drag coefficient
//         const double C_D = 24.0/Re * pow(1.0 + 0.27*Re, 0.43) + 
//                           0.47 * (1.0 - exp(-0.04 * pow(Re, 0.38)));
        
//         // Thermal velocity
//         const double v_th = sqrt(8.0 / M_PI) * cs;
        
//         // Drag coefficient term
//         const double C_D_term = 3.0/8.0 * v_rel_mag/v_th * C_D;
        
//         // Stopping time
//         const double ts_inv = gas_density / SOLID_RHO * v_th / R_PL * fmin(1.0, C_D_term);
        
//         // Safety check for stopping time
//         if (!isfinite(ts_inv) || ts_inv <= 0.0) continue;
        
//         const double ts = 1.0 / ts_inv;
        
//         // Acceleration due to gas drag
//         const double a_x = -1.0/ts * v_rel_x;
//         const double a_y = -1.0/ts * v_rel_y;
//         const double a_z = -1.0/ts * v_rel_z;
        
//         // Add acceleration to particle (check for finite values)
//         if (isfinite(a_x)) p->ax += a_x;
//         if (isfinite(a_y)) p->ay += a_y;
//         if (isfinite(a_z)) p->az += a_z;
//     }
// }

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "rebound.h"
#include "reboundx.h"

// Physical constants (in simulation units: Msun, AU, yr)
static const double R_CHAR = 40.0;
static const double JTOS = 0.000954588;
static const double R_PL = 6.684587122268445e-07;
static const double SOLID_RHO = 1683721.7643842339;
static const double MU = 2.33;
static const double M_PROTON = 8.411856872862986e-58;
static const double SIGMA_CHAR = 3.038845902395208e-07;
static const double LG = 0.00841860882192117;
static const double KB = 3.0898292661510003e-61;

void test_force(struct reb_simulation* const sim, struct rebx_force* const force, struct reb_particle* const particles, const int N) {
    
    const double G = sim->G;
    const double M_star = particles[0].m;
    const int N_active = sim->N_active;
    
    // Loop over test particles only
    for (int i = N_active; i < N; i++) {
        struct reb_particle* p = &particles[i];
        
        const double x = p->x;
        const double y = p->y;
        const double z = p->z;
        const double vx = p->vx;
        const double vy = p->vy;
        const double vz = p->vz;
        
        // Calculate radial distance
        const double R = sqrt(x*x + y*y);
        
        // Calculate orbital elements to get semimajor axis
        struct reb_orbit orb = reb_orbit_from_particle(G, *p, particles[0]);
        const double a = orb.a;
        
        // Safety checks
        if (a <= 0.0 || R <= 0.0) {
            continue; // Skip this particle if semimajor axis or radius is invalid
        }
        
        const double Omega = sqrt(G * M_star / (R*R*R));
        
        // Temperature profile (Bae et al. 2019)
        const double T = 44.0 * pow(R / 22.0, -0.24);
        
        // Sound speed and scale height
        const double cs = sqrt(KB * T / (MU * M_PROTON));
        const double H = cs / Omega;
        
        // Check for valid scale height
        if (H <= 0.0 || !isfinite(H)) {
            continue;
        }
        
        // Surface density profile
        double gas_surface_density = SIGMA_CHAR * pow(R / R_CHAR, -1.0) * exp(-R / R_CHAR);
        //double gas_surface_density = 0.1;
        // Gap between R=18 and R=40
        if (R >= 18.0 && R <= 40.0) {
            gas_surface_density *= 0.01;
        }
        
        // Vertical density profile (Gaussian)
        const double gas_density = gas_surface_density / (sqrt(2.0 * M_PI) * H) * 
                                   exp(-z*z / (2.0 * H*H));
        
        // Keplerian velocity at radius R
        const double v_K = sqrt(G * M_star / R);
        const double theta = atan2(y, x);
        
        // Gas velocity (azimuthal Keplerian flow)
        const double v_gas_x = -v_K * sin(theta);
        const double v_gas_y = v_K * cos(theta);
        const double v_gas_z = 0.0;
        
        // Relative velocity
        const double v_rel_x = v_gas_x - vx;
        const double v_rel_y = v_gas_y - vy;
        const double v_rel_z = v_gas_z - vz;
        const double v_rel_mag = sqrt(v_rel_x*v_rel_x + v_rel_y*v_rel_y + v_rel_z*v_rel_z);
        
        // Skip if relative velocity is too small
        if (v_rel_mag < 1e-60) {
            continue;
        }
        
        // FILE *fp = fopen('/data/ally/thesis_code/debug.log', 'a');
        // if (fp != NULL) {
        //     fprintf(fp, '%f', v_rel_mag);
        //     fclose(fp);
        // }

        // Reynolds number
        const double Re = 4.0 * R_PL * v_rel_mag / (cs * LG);
        
        // Check for valid Reynolds number
        if (Re <= 0.0 || !isfinite(Re)) {
            continue;
        }
        
        // Drag coefficient
        const double C_D = 24.0/Re * pow(1.0 + 0.27*Re, 0.43) + 
                          0.47 * (1.0 - exp(-0.04 * pow(Re, 0.38)));
        
        // Thermal velocity
        const double v_th = sqrt(8.0 / M_PI) * cs;
        
        // Drag coefficient term
        const double C_D_term = 3.0/8.0 * v_rel_mag/v_th * C_D;
        
        // Stopping time
        const double ts_inv = gas_density / SOLID_RHO * v_th / R_PL * fmin(1.0, C_D_term);
        
        // Check for valid stopping time
        if (ts_inv <= 0.0 || !isfinite(ts_inv)) {
            continue;
        }
        
        const double ts = 1.0 / ts_inv;
        
        // Acceleration due to gas drag
        const double a_x = 1.0/ts * v_rel_x;
        const double a_y = 1.0/ts * v_rel_y;
        const double a_z = 1.0/ts * v_rel_z;
        
        // Add acceleration to particle
        p->ax += a_x;
        p->ay += a_y;
        p->az += a_z;
    }
}
