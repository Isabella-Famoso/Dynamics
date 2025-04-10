import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as anim
from pandas import read_csv
from tqdm import tqdm
from scipy.integrate import quad
from scipy.signal import argrelmax

#To view the animations type %matplotlib qt6 in the console

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

#%%----Simulation and analysis code properties----

N = 30000 #Number of particles
M = 30000 #Total mass in solar masses
b = 250 #Total radius in AU
m_i = M / N #Particle mass in solar masses
equil = True #True if the generated distribution starts in equilibrium
display_animation = False #True to display the distribution 3D animation

#Relative path of the input file
if equil == True:
    path = "Equilibrium/PS_N" + str(N) + "_M" + str(M) + "_b" + str(b)
else:
    path = "PS_N" + str(N) + "_M" + str(M) + "_b" + str(b)

#%%----GetData functions----

#Function that reads the simulation data from the output file

def GetData(filename, N):
    #Get data in a pandas dataframe
    data = read_csv(filename, 
                    names=["x", "y", "z"],
                    sep="\s+")
    
    #Get the simulation time (first column of the corresponding rows)
    time = np.array(data[2::3*N + 3])[:, 0]
    
    #Get the particle masses
    m = np.unique(np.array(data[3:N+3])[:, 0])

    #Remove the rows corresponding to N and time (which are padded with NaNs)
    data = data.dropna().reset_index(drop=True)
    
    #Get the particle coordinates and velocities in the external frame
    #Rows are the evolution in time of a given particle, columns are the particle
    #number, and the third dimension are its coordinates/velocities
    pos_ext = np.array([data[i::2*N] for i in range(N)]).transpose(1, 0, 2)
    vel_ext = np.array([data[i+N::2*N] for i in range(N)]).transpose(1, 0, 2)
    
    #Compute CM position ad velocity
    CM_p = m * np.sum(pos_ext, axis=1) / (m*N)
    CM_v = m * np.sum(vel_ext, axis=1) / (m*N)
    
    #Convert from external to CM frame
    pos_CM = pos_ext.copy()
    vel_CM = vel_ext.copy()
    
    for i in range(CM_p.shape[0]):
        pos_CM[i, :, :] -= CM_p[i, :]
        vel_CM[i, :, :] -= CM_v[i, :]
    
    return time, m[0], pos_ext, vel_ext, pos_CM, vel_CM, CM_p, CM_v


#Function that reads the data from the log file

def GetLogData(filename): 
    with open(filename, "r") as log_file:
        output = np.array(log_file.read().splitlines())        
    
        idx = np.where(output == output[9])[0] #Find the row with T and U
        
    K_tot = np.zeros(len(idx))    
    U_tot = np.zeros(len(idx))   
        
    for i, j in enumerate(idx):
        K_tot[i] = float(output[j + 1].split()[1])
        U_tot[i] = -float(output[j + 1].split()[2])
    
    return K_tot, U_tot

#%%----Read output file----

#Get the simulation time and the positions and velocities of the particles

sim_file = "Output/" + path + ".out"

time, m_i, pos_ext, vel_ext, pos_CM, vel_CM, CM_p, CM_v = GetData(sim_file, N)

#Time in years
time_yr = time / t_iu_yr

#Positions and velocities of the particles in the external frame
x_ext, y_ext, z_ext = pos_ext[:, :, 0], pos_ext[:, :, 1], pos_ext[:, :, 2]
vx_ext, vy_ext, vz_ext = vel_ext[:, :, 0], vel_ext[:, :, 1], vel_ext[:, :, 2]

R_ext = np.sqrt(np.sum(pos_ext**2, axis=2))
V_ext = np.sqrt(np.sum(vel_ext**2, axis=2))
            
#Positions and velocities of the particles in the CM frame
x_CM, y_CM, z_CM = pos_CM[:, :, 0], pos_CM[:, :, 1], pos_CM[:, :, 2]
vx_CM, vy_CM, vz_CM = vel_CM[:, :, 0], vel_CM[:, :, 1], vel_CM[:, :, 2]

R_CM = np.sqrt(np.sum(pos_CM**2, axis=2))
V_CM = np.sqrt(np.sum(vel_CM**2, axis=2))

#Position and velocity of CM in the external frame
CM_x, CM_y, CM_z = CM_p[:, 0], CM_p[:, 1], CM_p[:, 2]
CM_vx, CM_vy, CM_vz = CM_v[:, 0], CM_v[:, 1], CM_v[:, 2]

CM_R = np.sqrt(np.sum(CM_p**2, axis=1))
CM_V = np.sqrt(np.sum(CM_v**2, axis=1))

#%%----Plot the velocity distribution----

if equil == True:       
    #Theoretical velocity distribution
    norm_q = quad(lambda q: (1 - q**2)**(7/2) * q**2, 0, 1)[0]
    v_d = lambda q: (1 - q**2)**(7/2) * q**2 / norm_q
    
    #Escape velocity: v = q * v_e
    v_e = np.sqrt(2 * M / np.sqrt(R_CM**2 + b**2)) 
    
    fig_vel, ax_vel = plt.subplots()
    bbox_vel = dict(boxstyle='round', fc='white', ec='black', alpha=0.4)

    def update_vel(frame):
        ax_vel.clear()
        
        ax_vel.hist((V_CM / v_e)[frame, :], bins="fd", color="darkcyan", density=True, label="Generated velocities")    
        ax_vel.plot(np.sort(V_CM / v_e)[frame, :], v_d(np.sort(V_CM / v_e)[frame, :]), color="crimson", label="Theoretical distribution")
        ax_vel.set_xlabel("$v / v_e$")
        ax_vel.legend()
        ax_vel.text(0.03, 0.79, s="t =" + f"{time_yr[frame]: .3f}" + " yr", bbox=bbox_vel, color="black", size=10, transform=ax_vel.transAxes)
        
    end_vel = np.where(time == max(time))[0][0]
    animation_vel = anim.FuncAnimation(fig=fig_vel, 
                                   func=update_vel, 
                                   frames=end_vel, 
                                   interval=120, repeat=True)

#%%----Compute total energy in the CM frame----

#Compute the total potential and kinetic energy of the system in the CM frame

K_tot, U_tot = GetLogData("../Plummer Sphere/Logs/" + path + "_log")
K_tot *= M_sun * v_iu_cgs**-2
U_tot *= G_p * M_sun**2 * AU**-1

CM_K = 1/2 * m_i * CM_V**2
K_CM = K_tot - CM_K

E_tot_CM = K_CM + U_tot

plt.figure()
plt.plot(time_yr, K_CM, color="crimson", label="Total kinetic energy")
plt.plot(time_yr, U_tot, color="darkcyan", label="Total potential energy")
plt.plot(time_yr, E_tot_CM, color="green", label="Total energy")
plt.xlabel("$t\ [yr]$")
plt.ylabel("$E\ [erg]$")
plt.legend()

#%%----Compute total linear and angular momentum in the CM and external frames----

p_tot_ext = m_i * M_sun * np.sqrt(np.sum(vx_ext, axis=1)**2 + 
                              np.sum(vy_ext, axis=1)**2 + 
                              np.sum(vz_ext, axis=1)**2) / v_iu_cgs
L_tot_ext = m_i * M_sun * AU * v_iu_cgs**-1 * np.sqrt(np.sum(y_ext * vz_ext - z_ext * vy_ext, axis=1)**2 + 
                                  np.sum(z_ext * vx_ext - x_ext * vz_ext, axis=1)**2 + 
                                  np.sum(x_ext * vy_ext - y_ext * vx_ext, axis=1)**2)

p_tot_CM = m_i * M_sun * np.sqrt(np.sum(vx_CM, axis=1)**2 + 
                              np.sum(vy_CM, axis=1)**2 + 
                              np.sum(vz_CM, axis=1)**2) / v_iu_cgs
L_tot_CM = m_i * M_sun * AU * v_iu_cgs**-1 * np.sqrt(np.sum(y_CM * vz_CM - z_CM * vy_CM, axis=1)**2 + 
                                  np.sum(z_CM * vx_CM - x_CM * vz_CM, axis=1)**2 + 
                                  np.sum(x_CM * vy_CM - y_CM * vx_CM, axis=1)**2)

fig_p, ax_p = plt.subplots(2, 2, figsize=(10, 10))

ax_p[0, 0].plot(time_yr, p_tot_ext, color="darkcyan")
ax_p[0, 0].set_ylabel("$p_{ext}\ [g\cdot cm\cdot s^{-1}]$")
ax_p[0, 0].set_ylim(0.9*np.min(p_tot_ext), 1.1*np.max(p_tot_ext))

ax_p[0, 1].plot(time_yr, L_tot_ext, color="darkcyan")
ax_p[0, 1].set_ylabel("$L_{ext}\ [g\cdot cm^{2}\cdot s^{-1}]$")
ax_p[0, 1].set_ylim(0.9*np.min(L_tot_ext), 1.1*np.max(L_tot_ext))

ax_p[1, 0].plot(time_yr, p_tot_CM, color="darkcyan")
ax_p[1, 0].set_ylabel("$p_{CM}\ [g\cdot cm\cdot s^{-1}]$")
ax_p[1, 0].set_ylim(0.9*np.min(p_tot_CM), 1.1*np.max(p_tot_CM))

ax_p[1, 1].plot(time_yr, L_tot_CM, color="darkcyan")
ax_p[1, 1].set_ylabel("$L_{CM}\ [g\cdot cm^{2}\cdot s^{-1}]$")
ax_p[1, 1].set_ylim(0.9*np.min(L_tot_CM), 1.1*np.max(L_tot_CM))

fig_p.supxlabel("$t\ [yr]$", y=0.06)

#%%----Compute theoretical timescales----

#Compute the dynamical time in years

density_0 = M * M_sun / (4/3 * np.pi * (np.max(R_CM) * AU)**3)
t_dyn = np.sqrt(3 * np.pi / (16 * G_p * density_0)) / year

if equil == True:
    #Compute relaxation timescale
    
    n_rel = N / (8 * np.log(N))
    t_cross = np.median(R_CM[0, :]) / np.median(V_CM[0, :])
    t_rel = t_cross * n_rel
 
    print("\nCrossing time: t_cross =" + f"{t_cross / t_iu_yr: .3f}" + " yr =" + f"{t_cross: .3f}" + " IU")
    print("Relaxation time: t_rel =" + f"{t_rel / t_iu_yr: .3f}" + " yr =" + f"{t_rel: .3f}" + " IU")
    
else:
    #Compute collapse timescale
    t_coll = t_dyn / np.sqrt(2)
    print("Collapse time: t_coll =" + f"{t_coll: .3f}" + " yr =" + f"{t_coll * t_iu_yr: .3f}" + " IU")


print("Dynamical time: t_dyn =" + f"{t_dyn: .3f}" + " yr =" + f"{t_dyn * t_iu_yr: .3f}" + " IU")
print("Initial mean density =" + f"{density_0: .3e}" + " g cm^-3")

#%%----Compute simulation timescales----
#%%%Compute the collapse time from the simulation data

if equil == False:
    #Index of the particle with the farthest position from the CM at initial time
    initial_R_idx = np.where(R_CM[0, :] == np.max(R_CM[0, :]))[0][0]
    
    #Index of the minimum value in time of the radius of the farthest particle
    min_R_idx = np.where(R_CM[:, initial_R_idx] == np.min(R_CM[:, initial_R_idx]))[0][0]
    
    #Collapse time from the simulation
    t_coll_sim = time[min_R_idx]
    
    print("Collapse time from simulation =" + f"{t_coll_sim / t_iu_yr: .3f}" + " yr")

#%%%Compute the collapse time from the point of minimum potential energy

if equil == False:
    t_coll_en = time[U_tot == np.min(U_tot)][0]

    print("Collapse time from minimum U_tot =" + f"{t_coll_en / t_iu_yr: .3f}" + " yr")
#%%----Simulation animation----

if display_animation:
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(projection="3d")
    ax.set_aspect("equal")
    
    ticks = np.linspace(-2, 2., 5)
    bbox = dict(boxstyle='round', fc='white', ec='black', alpha=0.5)
    
    def update_anim(frame):
        ax.clear()
        
        ax.set_autoscale_on(False)
        ax.scatter(x_CM[frame], y_CM[frame], z_CM[frame], color="darkcyan")
        ax.scatter(CM_x[frame], CM_y[frame], CM_z[frame], color="crimson")
        ax.set_xticks(ticks)
        ax.set_yticks(ticks)
        ax.set_zticks(ticks)
        ax.set_xlabel("x [AU]", size=10)
        ax.set_ylabel("y [AU]", size=10)
        ax.set_zlabel("z [AU]", size=10)
        ax.text(0.875, 0.04, 0., s="t =" + f"{time_yr[frame]: .3f}" + " yr", bbox=bbox, color="black", size=10, transform=ax.transAxes)
        
    end = np.where(time == max(time))[0][0]
    animation = anim.FuncAnimation(fig=fig, 
                                   func=update_anim, 
                                   frames=end, 
                                   interval=20, repeat=True)

#%%----Phase space visualization----

fig_ph = plt.figure()

#Phase space plot
ax_ph = fig_ph.add_subplot()    
bbox_ph = dict(boxstyle='round', fc='white', ec='black', alpha=0.8)

def update_ph(frame):
    ax_ph.clear()
    
    #Show only particles that start close to the center
    close_particles = R_CM[frame, :] < 2 * b
    
    ax_ph.hist2d(R_CM[frame, close_particles], V_CM[frame, close_particles], bins=int(np.sqrt(np.sum(close_particles))))
    ax_ph.set_xlabel("R [AU]", size=10)
    ax_ph.set_ylabel("V [IU]", size=10)   
    ax_ph.set_ylim(top=np.max(V_CM[0, close_particles]))
    ax_ph.text(0.79, 0.92, s="t =" + f"{time_yr[frame]: .3f}" + " yr", bbox=bbox_ph, color="black", size=10, transform=ax_ph.transAxes)
    
    
end_ph = np.where(time == max(time))[0][0]
animation_ph = anim.FuncAnimation(fig=fig_ph, 
                               func=update_ph, 
                               frames=end_ph, 
                               interval=60, repeat=True)

#%%----Orbits of particles----

if equil == True:
    #Choose random particles
    # np.random.seed(5)
    
    choice_size = 1
    choice_idx =  np.random.choice(np.arange(0, N - 1), replace=False, size=choice_size)
    
    #Phi coordinate of the particle
    phi_orbit = np.arctan(y_CM[:, choice_idx] / x_CM[:, choice_idx])
    
    #Plot the chosen particles radius over time
    fig_coord, ax_coord = plt.subplots(1, 2, figsize=(10, 5))
    
    for i in choice_idx:
        ax_coord[0].plot(time_yr, R_CM[:, i], color="darkcyan")
        ax_coord[1].plot(time_yr, phi_orbit[:], color="darkcyan")
        
    fig_coord.supxlabel("$t\ [yr]$")
    ax_coord[0].set_ylabel("$R\ [AU]$")
    ax_coord[1].set_ylabel("$\phi\ [rad]$")
    
    #Plot the chosen particles orbits in 3D
    
    fig_o = plt.figure(figsize=(10, 10))
    ax_o = fig_o.add_subplot(projection="3d")
    
    for i in choice_idx:
        ax_o.plot(x_CM[:, i], y_CM[:, i], z_CM[:, i])
        
    ax_o.set_xlabel("$x\ [AU]$")
    ax_o.set_ylabel("$y\ [AU]$")
    ax_o.set_zlabel("$z\ [AU]$")
    

    #Compute the periods from the maxima of the R and phi coordinates
    
    #The apocenter and pericenter are the extrema of the radius plot
    # R_orbit_grad = np.gradient(R_CM[:, choice_idx], axis=0)
    # R_max_points_idx = np.where(np.isclose(R_orbit_grad, 0, atol=1e-1) == True)[0]
    R_max_points_idx = np.array(argrelmax(R_CM[:, choice_idx])[0])
    R_max_points = time_yr[R_max_points_idx]
    
    #In the phi plot the max and min have a higher derivative than the other points
    phi_orbit_grad = np.gradient(phi_orbit, axis=0)
    phi_max_points_idx = np.where(phi_orbit_grad > 1e-1)[0]
    phi_max_points = time_yr[phi_max_points_idx][1::2]  #The max is after the min
    
    #Periods
    T_r = np.median(np.diff(R_max_points))
    T_phi = np.median(np.diff(phi_max_points))
    
    #Interval of the angle between apocenter and pericenter, used in the theory period
    dphi = np.abs(np.median(np.diff(phi_orbit[R_max_points_idx], axis=0)))
    
    print("T_R = " + str(T_r) + " yr")
    print("T_dphi = " + str(T_phi) + " yr")
    print("Theoretical T_phi = " + str(2*np.pi/dphi * T_r) + " yr")
    print("T_dphi/T_R = " + str(2*np.pi/dphi))

#%%----Plot the distribution density over time----

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
p_num = np.zeros((len(time), len(r_grid) - 1))

for i in range(0, len(time)):
    #Count the number of particles inside every radius interval at a given time
    p_idx, p_inside = np.unique(np.digitize(R_CM[i, :], r_grid), return_counts=True)
    
    #Count only the particles within a, and add the number to the p_num
    #elements at the index of the corresponding occupied interval, leaving
    #the unoccupied ones at zero particles    
    cond_within = np.where(p_idx < len(r_grid) - 1)
    p_num[i, p_idx[cond_within]] += p_inside[cond_within]

#Compute the density in each radius interval and the Poisson error
rho = (p_num / volume) * m_i * M_sun * AU**-3
rho_err = (np.sqrt(p_num) / volume) * m_i * M_sun * AU**-3

#%%%Density profile animation

#Theoretical density profile
r_range = np.linspace(np.min(R_CM), 3 * b, len(r_grid))
rho_th = lambda r: 3 * M / (4 * np.pi* b**3) * (1 + r**2/b**2)**(-5/2) * M_sun * AU**-3

#Plot the density at each radius corresponding to the middle of a bin
bar_pos = np.diff(r_grid) / 2 + r_grid[:-1]

fig_rho, ax_rho = plt.subplots()
bbox_rho = dict(boxstyle='round', fc='white', ec='black', alpha=0.4)

def update_rho(frame):
    ax_rho.clear()
    
    ax_rho.plot(bar_pos, rho[frame, :], color="darkcyan", label="Simulation density profile")
    ax_rho.plot(r_range, rho_th(r_range), color="crimson", label="Analytical density profile")
    ax_rho.plot(r_range, 3 * M / (4 * np.pi* b**-2) * r_range**-5 * M_sun * AU**-3, alpha=0.3, linestyle="--", color="green", label="$\\rho(r) \propto r^{-5}$")
    ax_rho.axhline(3 * M / (4 * np.pi* b**3) * M_sun * AU**-3, alpha=0.3, linestyle="-.", color="green", label="$\\rho(r) = const.$")
    ax_rho.axvline(b, alpha=0.6, color="grey", linestyle="--", label="$r = b$")
    ax_rho.fill_between(bar_pos, 
                      rho[frame, :] - rho_err[frame, :], 
                      rho[frame, :] + rho_err[frame, :], alpha=0.3, color="darkcyan", label="$1\sigma$ Poisson error")
    ax_rho.set_xlabel("$R\ [AU]$")
    ax_rho.set_ylabel("$\\rho\ [g\ cm^{-3}]$")
    ax_rho.set_ylim(0, 1.2 * 3 * M / (4 * np.pi* b**3) * M_sun * AU**-3)
    ax_rho.legend(loc="upper right")
    ax_rho.text(0.81, 0.53, s="t =" + f"{time_yr[frame]: .3f}" + " yr", bbox=bbox_rho, color="black", size=10, transform=ax_rho.transAxes)
    
end_rho = np.where(time == max(time))[0][0]
animation_rho = anim.FuncAnimation(fig=fig_rho, 
                               func=update_rho, 
                               frames=end_rho, 
                               interval=120, repeat=True)

#%%----Plot the potential profile over time----

#Compute the potential on a r_grid on the equatorial plane (phi = theta = 0)
#For spherical symmetry this is the same potential for every other phi and theta
V = np.zeros((len(time), len(r_grid)))

for t in range(len(time)):
    for r in range(len(r_grid)):
        V[t, r] = -G_p * m_i * M_sun * AU**-1 * np.sum(np.sqrt((x_CM[t, :] - r_grid[r])**2 + 
                                                    y_CM[t, :]**2 + z_CM[t, :]**2)**-1)
        
#%%%Potential profile animation

#Theoretical potential profile
V_th = lambda r: -G_p * M * M_sun * AU**-1 * (r**2 + b**2)**(-1/2)

fig_V, ax_V = plt.subplots()
bbox_V = dict(boxstyle='round', fc='white', ec='black', alpha=0.4)

def update_V(frame):
    ax_V.clear()
    
    ax_V.plot(r_grid, V[frame, :], color="darkcyan", label="Simulation potential profile")
    ax_V.plot(r_range, V_th(r_range), color="crimson", label="Analytical potential profile")
    ax_V.axvline(b, alpha=0.6, color="grey", linestyle="--", label="$r = b$")
    ax_V.set_xlabel("$R\ [AU]$")
    ax_V.set_ylabel("$\Phi (R) \ [cm^2 \cdot s^{-2}]$")
    ax_V.legend(loc="lower right")
    ax_V.text(0.8, 0.23, s="t =" + f"{time_yr[frame]: .3f}" + " yr", bbox=bbox_V, color="black", size=10, transform=ax_V.transAxes)
    
end_V = np.where(time == max(time))[0][0]
animation_V = anim.FuncAnimation(fig=fig_V, 
                               func=update_V, 
                               frames=end_V, 
                               interval=120, repeat=True)

#%%----Compute the distribution function----

#Energy of the single particles
K_i = 0.5 * m_i * V_CM**2 * M_sun * v_iu_cgs**-2
U_i = np.zeros((len(time), N))

dpos = pos_CM.copy()

for t in tqdm(range(len(time))):
    for i in tqdm(range(N)):   
        # idx = np.concatenate((np.arange(0, i), np.arange(i + 1, N)))
        
        for j in range(3):
            dpos[t, :i, j] -= dpos[t, i, j]
            dpos[t, i+1:, j] -= dpos[t, i, j]
            
        U_i[t, :i] += -G_p * (m_i * M_sun)**2 * AU**-1 * np.sqrt(dpos[t, :i, 0]**2 +
                                                          dpos[t, :i, 1]**2 +
                                                          dpos[t, :i, 2]**2)**-1
        U_i[t, i+1:] += -G_p * (m_i * M_sun)**2 * AU**-1 * np.sqrt(dpos[t, i+1:, 0]**2 +
                                                          dpos[t, i+1:, 1]**2 +
                                                          dpos[t, i+1:, 2]**2)**-1
        
        for j in range(3):
            dpos[t, :i, j] += dpos[t, i, j]
            dpos[t, i+1:, j] += dpos[t, i, j]

E_i = K_i + U_i

#%%%Plot the distribution function

#Theoretical distribution function
norm = np.array([quad(lambda E: E**(7/2), 0, np.max(-E_i[t, :]))[0] for t in range(len(time))])
f_E = lambda E: E**(7/2)

fig_f, ax_f = plt.subplots()
bbox_f = dict(boxstyle='round', fc='white', ec='black', alpha=0.4)

def update_f(frame):
    ax_f.clear()
    
    ax_f.hist(-E_i[frame, :], bins="fd", color="darkcyan", density=True, label="Energy per particle distribution")    
    ax_f.plot(np.sort(-E_i[frame, :]), f_E(np.sort(-E_i[frame, :])) / norm[frame], color="crimson", label="$f(E) \propto (-E)^{7/2}$")
    ax_f.set_xlabel("$-E\ [erg]$")
    ax_f.set_ylabel("$f(E)\ [cm^{-3}]$")
    ax_f.legend(loc="upper left")
    ax_f.text(0.03, 0.79, s="t =" + f"{time_yr[frame]: .3f}" + " yr", bbox=bbox_f, color="black", size=10, transform=ax_f.transAxes)
    
end_f = np.where(time == max(time))[0][0]
animation_f = anim.FuncAnimation(fig=fig_f, 
                               func=update_f, 
                               frames=end_f, 
                               interval=120, repeat=True)