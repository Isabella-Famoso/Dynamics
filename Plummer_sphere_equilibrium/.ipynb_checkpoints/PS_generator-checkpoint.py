import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import quad
from tqdm import tqdm

#Conversion in IU

G_p = 6.67259e-8 #G in cgs
M_sun = 1.9891e33 #solar mass in g
R_sun = 6.9598e10 #solar radius in cm 
year = 3.14159e7
ly = 9.463e17 #light year in cm
parsec = 3.086e18 #parsec in cm
AU = 1.496e13 #astronomical unit in cm

def v_IU(v_p, M_p=M_sun, r_p=AU):
    return np.sqrt(r_p / (G_p * M_p)) * v_p

def t_IU(t_p, M_p=M_sun, r_p=AU):
    return t_p / (np.sqrt(r_p / (G_p * M_p)) * r_p)

t_iu_yr = t_IU(year) #1 yr is 6.251839 IU
v_iu_cgs = v_IU(1) #1 cm/s is 3.357e-7 IU

#%%----Simulation properties----

N = 20000 #Number of particles
M = 10000 #Total mass in solar masses
b = 200 #Total radius in AU
m_i = M / N #Particle mass in solar masses
equil = True #True if the generated distribution will start in equilibrium
generate = False #True to write the generated distribution to file

#Relative path of the generated file
if equil == True:
    path = "Equilibrium/PS_N" + str(N) + "_M" + str(M) + "_b" + str(b)
else:
    path = "PS_N" + str(N) + "_M" + str(M) + "_b" + str(b)

#%%----GenInput function----

#Function to generate a distribution of particles to input in the Treecode

def GenInput(N, m_i, a, to_file=False):
    
    #Cumulative distributions
    P_r = np.random.uniform(0, 1, N) 
    P_phi = np.random.uniform(0, 1, N)
    P_theta = np.random.uniform(0, 1, N)
    
    #Draw spherical coordinates based on their PDF using the inverse cumulative
    r = b * (P_r**(-2/3) - 1)**(-1/2)
    phi = 2 * np.pi * P_phi
    theta = np.arccos(1 - 2 * P_theta)
    
    #Convert in cartesian coordinates
    x_0 = r * np.cos(phi) * np.sin(theta)
    y_0 = r * np.sin(phi) * np.sin(theta)
    z_0 = r * np.cos(theta) 
    
    #Initial velocities
    
    if equil == True:
        psi_IU = -M / np.sqrt(r**2 + b**2)
        
        f = lambda q: (1 - q**2)**(7/2) * q**2
        max_f = np.max(f(np.linspace(0, 1, 1000)))
        
        q_samples = []
        
        while len(q_samples) < N:
            q_rand = np.random.uniform(0, 1)
            y = np.random.uniform(0, max_f)
            
            if y < f(q_rand):
                q_samples.append(q_rand)
                
        v = q_samples * np.sqrt(-2 * psi_IU)
        
        v_phi = 2 * np.pi * np.random.uniform(0, 1, N)
        v_theta = np.arccos(1 - 2 * np.random.uniform(0, 1, N))
    
        vx_0 = v * np.cos(v_phi) * np.sin(v_theta)
        vy_0 = v * np.sin(v_phi) * np.sin(v_theta)
        vz_0 = v * np.cos(v_theta)  
    
    else:
        vx_0 = np.zeros(N)
        vy_0 = np.zeros(N)
        vz_0 = np.zeros(N)
    
    #Write to file
    if to_file:
        file_path = "Input/" + path + ".in"
        
        with open(file_path, "w") as i_file:
            i_file.write(str(N) + "\n3\n0\n")
            
            for i in tqdm(range(N)):
                i_file.write(str(m_i) + "\n") 
                
            for i in tqdm(range(N)):
                i_file.write(str(x_0[i]) + " " + str(y_0[i]) + " " + str(z_0[i]) + "\n")
            
            for i in tqdm(range(N)):    
                i_file.write(str(vx_0[i]) + " " + str(vy_0[i]) + " " + str(vz_0[i]) + "\n")
        
    return x_0, y_0, z_0, vx_0, vy_0, vz_0, r, phi, theta
    
#%%----Generate input----

x_0, y_0, z_0, vx_0, vy_0, vz_0, r, phi, theta = GenInput(N, m_i, b, generate)

#%%%Softening parameter and command to run the Treecode

#Softening parameter epsilon
vol = 4/3 * np.pi * np.max(r)**3
eps = 1e-4 * (vol / N)**(1/3)

#Print command to run the Treecode
print('./treecode in="../Plummer Sphere/Input/' + path + '.in"' + 
      ' out="../Plummer Sphere/Output/' + path + '.out"' +
      ' dtime=1 eps=' + str(eps) + ' theta=1 tstop=0.01 dtout=0.01' + 
      ' > "../Plummer Sphere/Logs/' + path + '_log"')

#%%----Compute theoretical timescales----

#Compute the dynamical time in years

density_0 = M * M_sun / (4/3 * np.pi * (b * AU)**3)
t_dyn = np.sqrt(3 * np.pi / (16 * G_p * density_0)) / year

if equil == True:
    #Compute relaxation timescale
    
    n_rel = N / (8 * np.log(N))
    t_cross = np.median(r) / np.median(np.sqrt(vx_0**2 + vy_0**2 + vz_0**2))
    t_rel = t_cross * n_rel

    print("\nCrossing time: t_cross =" + f"{t_cross / t_iu_yr: .3f}" + " yr =" + f"{t_cross: .3f}" + " IU")
    print("Relaxation time: t_rel =" + f"{t_rel / t_iu_yr: .3f}" + " yr =" + f"{t_rel: .3f}" + " IU")
    
else:
    #Compute collapse timescale
    t_coll = t_dyn / np.sqrt(2)
    print("Collapse time: t_coll =" + f"{t_coll: .3f}" + " yr =" + f"{t_coll * t_iu_yr: .3f}" + " IU")


print("Dynamical time: t_dyn =" + f"{t_dyn: .3f}" + " yr =" + f"{t_dyn * t_iu_yr: .3f}" + " IU")
print("Initial mean density within b = " + f"{density_0: .3e}" + " g cm^-3")

#%%----Visualize the generated distribution----

fig = plt.figure(figsize=(10, 10))
ax = fig.add_subplot(projection="3d")
ax.set_aspect("equal")
ax.scatter(x_0, y_0, z_0, alpha=0.8, color="darkcyan")

#%%----Plot the spherical coordinates distributions----

fig_sp, ax_sp = plt.subplots(1, 3, figsize=(10, 5))

ax_sp[0].hist(np.log10(r), bins="fd", color="darkcyan")
ax_sp[0].set_xlabel("$\log{r}\ [AU]$")

ax_sp[1].hist(phi, bins="fd", color="darkcyan")
ax_sp[1].set_xlabel("$\phi\ [rad]$")

ax_sp[2].hist(theta, bins="fd", color="darkcyan")
ax_sp[2].set_xlabel("$\theta\ [rad]$")

#%%----Plot the radial velocity distribution----

if equil == True:    
    #Velocity modulus
    v = np.sqrt(vx_0**2 + vy_0**2 + vz_0**2)
    
    fig_sp = plt.figure()
    
    plt.hist(v, bins="fd", color="darkcyan", density=True)
    plt.xlabel("$v_{r}\ [IU]$")
    
#%%----Phase space visualization----

if equil == True:
    #Show only particles that start close to the center
    close_particles = r < 2 * b
    
    plt.figure()
    plt.hist2d(r[close_particles], v[close_particles], bins=80)
    plt.xlabel("R [AU]", size=10)
    plt.ylabel("V [IU]", size=10)   

#%%----Compute the density distribution----

#Function to divide the distribution in intervals
def DivideDistribution(edge, r_grid_len, equal_volume=False):   
    if equal_volume:
        #Divide the sphere in volume elements, all with the same total volume
        volume = 4/3 * np.pi * edge**3 / r_grid_len

        #Compute the radii of each volume element to form a grid of radii, up to an edge
        r_grid = np.zeros(r_grid_len)

        for i in range(r_grid_len - 1):
            r_grid[i + 1] = (3/4 / np.pi * volume + r_grid[i]**3)**(1/3)
            
        if r_grid[-1] < edge:
            r_grid = np.append(r_grid, edge)
            
    else:
        #Divide the sphere in volume elements, with constant radius intervals
        r_grid = np.linspace(0, edge, r_grid_len)
            
        if r_grid[-1] < edge:
            r_grid = np.append(r_grid, edge)
            
        volume = 4/3 * np.pi * np.array([r_grid[i + 1]**3 - r_grid[i]**3 for i in range(len(r_grid) - 1)])

    return r_grid, volume


#Divide the distribution in radius intervals
r_grid, volume = DivideDistribution(2 * b, 250, False)

#Compute the number of particles inside a given radius interval    
p_num = np.zeros(len(r_grid) - 1)

#Count the number of particles inside every radius interval at a given time
p_idx, p_inside = np.unique(np.digitize(r, r_grid), return_counts=True)

#Count only the particles within a, and add the number to the p_num
#elements at the index of the corresponding occupied interval, leaving
#the unoccupied ones at zero particles    
cond_within = np.where(p_idx < len(r_grid) - 1)
p_num[p_idx[cond_within]] += p_inside[cond_within]

#Compute the density in each radius interval and the Poisson error
rho = (p_num / volume) * m_i * M_sun * AU**-3
rho_err = (np.sqrt(p_num) / volume) * m_i * M_sun * AU**-3

#%%%Plot the density at each radius corresponding to the middle of a bin

#Theoretical density profile
r_range = np.arange(np.min(r), 3*b)
rho_th = lambda r: 3 * M / (4 * np.pi* b**3) * (1 + r**2/b**2)**(-5/2) * M_sun * AU**-3

#Center of every interval
bar_pos = np.diff(r_grid) / 2 + r_grid[:-1]

fig_rho, ax_rho = plt.subplots()
ax_rho.plot(bar_pos, rho, color="darkcyan", label="Density profile")
ax_rho.plot(r_range, rho_th(r_range), color="crimson", label="Analytical density profile")
ax_rho.plot(r_range, 3 * M / (4 * np.pi* b**-2) * r_range**-5 * M_sun * AU**-3, alpha=0.3, linestyle="--", color="green", label="$\\rho(r) \propto r^{-5}$")
ax_rho.axhline(3 * M / (4 * np.pi* b**3) * M_sun * AU**-3, alpha=0.3, linestyle="-.", color="green", label="$\\rho(r) = const.$")
ax_rho.axvline(b, alpha=0.6, color="grey", linestyle="--", label="$R = b$")
ax_rho.fill_between(bar_pos, 
                  rho - rho_err, 
                  rho + rho_err, alpha=0.3, color="darkcyan", label="$1\sigma$ Poisson error")
plt.xlabel("$R\ [AU]$")
plt.ylabel("$\\rho\ [g\ cm^{-3}]$")
plt.ylim(0, 1.2 * 3 * M / (4 * np.pi* b**3) * M_sun * AU**-3)
plt.legend()

#%%----Mean density of the sphere at various radii----

#Particle radii
R = np.sqrt(x_0**2 + y_0**2 + z_0**2)

#Sphere number density and matter density inside a shell at radius R_i
n_density = np.array([np.sum(R <= R_i) / (4/3 * np.pi * R_i**3) for R_i in np.unique(np.sort(R))])
n_density_err = np.array([np.sqrt(np.sum(R <= R_i)) / (4/3 * np.pi * R_i**3) for R_i in np.unique(np.sort(R))])
rho_mean = m_i * M_sun * n_density * AU**-3
rho_mean_err = m_i * M_sun * n_density_err * AU**-3

#Theoretical mean density
rho_mean_th = lambda r: M * r**3 / (b**3 * (1 + r**2 / b**2)**(3/2)) / (4/3 * np.pi * r**3) * M_sun * AU**-3

fig_n, ax_n = plt.subplots()
ax_n.plot(np.unique(np.sort(R)), rho_mean, color="darkcyan", label="Density within a given radius")
ax_n.plot(np.unique(np.sort(R)), rho_mean_th(np.unique(np.sort(R))), color="crimson", label="Analytical density profile")
ax_n.fill_between(np.unique(np.sort(R)), 
                  rho_mean - rho_mean_err, 
                  rho_mean + rho_mean_err, alpha=0.3, color="darkcyan", label="$1\sigma$ Poisson error")
plt.xlabel("$R\ [AU]$")
plt.ylabel("$\\rho\ [g\ cm^{-3}]$")
plt.yscale("log")
plt.legend()

#%%----Plot the potential profile----

V = np.zeros(len(r_grid))

for r in range(len(r_grid)):
    V[r] = -G_p * m_i * M_sun * AU**-1 * np.sum(np.sqrt((x_0 - r_grid[r])**2 + 
                                                    y_0**2 + z_0**2)**-1)
        
#%%%Potential profile plot

#Theoretical profile
V_th = lambda r: -G_p * M * M_sun * AU**-1 * (r**2 + b**2)**(-1/2)
        
plt.figure()
plt.plot(r_grid, V, color="darkcyan", label="Simulation potential profile")
plt.plot(r_grid, V_th(r_grid), color="crimson", label="Theoretical potential profile")
plt.xlabel("$r\ [AU]$")
plt.ylabel("$\Phi (r)\ [cm^2 \cdot s^{-2}]$")
plt.legend()

