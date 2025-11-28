# %%
"""
Runnable Jupyter-style notebook (Python) for:
Centralized MPC on the IEEE 14-bus system using ANDES.


It performs the following high-level steps:
  0. Install & import required packages (andes, sippy, control, casadi, numpy, pandas, matplotlib)
  1. Load IEEE-14 case from ANDES and run baseline simulation with disturbances
  2. Collect time-series data (bus voltages, generator speeds, inputs)
  3. Identify a reduced-order state-space model (N4SID via sippy)
  4. Verify identified model by simulating its response to the same inputs
  5. Formulate and test a centralized MPC (CasADi) on the identified discrete model
  6. Implement an external discrete-time control loop that runs ANDES and applies MPC actions
  7. Validate closed-loop performance and plot results

Assumptions & caveats:
- The identified model is a reduced-order linear discrete-time system. The MPC is designed
  on that model and applied in closed-loop to the full nonlinear ANDES simulator via an
  external sample-and-hold control loop.

"""

# %%
# 0) Install required packages (run this cell first if running in a fresh environment)
# NOTE: comment out or modify installs if packages are already present or if you use conda.
 
# import sys
# import subprocess
# import pkgutil

# def pip_install(packages):
#     for pkg in packages:
#         if pkgutil.find_loader(pkg) is None:
#             print(f"Installing {pkg}...")
#             subprocess.check_call([sys.executable, '-m', 'pip', 'install', pkg])
#         else:
#             print(f"{pkg} already installed")

# required = ['andes-dyn', 'sippy', 'casadi', 'control', 'pandas', 'matplotlib', 'numpy', 'scipy']
# # Note: package names may vary (andes-dyn or andes). Adjust to your environment if needed.

# pip_install(required)

# %%
# 0) Imports
print("--------------------- Importing Required Packages ---------------------")
import data_cleaner
import numpy as np
import pandas as pd
# import matplotlib
# matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import time as _time

# ANDES import - adjust if your ANDES package name differs
try:
    import andes
except Exception as e:
    raise ImportError('Failed to import ANDES. Ensure ANDES is installed and importable (e.g. pip install andes-dyn).')

from sippy_unipi import system_identification as sysid
import casadi as ca
import cvxpy as cp
from control import ss, forced_response

from funcs import ANDES_Case_Runner, append_alter_rows

# %%
# 1) Load IEEE-14 and run an open-loop simulation to collect data
# The aim: excite both active and reactive dynamics with several disturbances
print("--------------------- 1. Running Dynamic Simulation ---------------------")

# case = 'kundur_full.xlsx'  
case = 'IEEE_cases/ieee14_alter_v4.xlsx' 
# case = 'IEEE_cases/ieee14_alter_v3.xlsx'  
# case = 'IEEE_cases/ieee14_alter_v0.xlsx'  
# case = 'IEEE_cases/ieee14_v0.xlsx'  

T_END = 20.0


# ssys = ANDES_Case_Runner(case, T_END) 
ssys = pd.read_csv('ieee14_alter_v4_out.csv')

# %%
# 2) Extract time-series data
print("--------------------- 2. Extracting Time-Series Data ---------------------")

time = ssys['Time [s]'].values
n_t = len(time)
print('Time samples:', n_t)
t_dt = data_cleaner.data_cleaner(time)

# Generator rotor speeds (omega) and bus voltages
# omega = ssys['omega GENROU 1']
# TS_Data.x[:, ssys.GENROU.omega.a]
# Vbus = TS_Data.y[:, ssys.Bus.v.a]
no_buses = 14
Vbus = ssys[[f'v Bus {b}' for b in range(1, no_buses+1)]].values
print('Bus voltage data:', Vbus)

# Choose outputs and inputs for identification & MPC

# Outputs: Select voltage measurements at a subset of buses (buses [2,4,7,9])
p = 4 # number of outputs (4 voltages)
buses_of_interest = [2,4,7,9]
V_meas = Vbus[:, [b-1 for b in buses_of_interest]]

plt.figure()
plt.plot(time, V_meas[:, 0], '.')
plt.title('Voltage of Bus 2')
plt.xlabel('Time (s)')
plt.grid(True)
# plt.savefig('IEEE14_TDS_Raw_V_Data.png')

# Inputs: construct a synthetic input vector representing the known load steps
# We'll approximate the load steps by reconstructing them from ANDES events: here we know
# dP at times 1,4,8 for buses 2,6,9 respectively
m = 3 # number of inputs
u = np.zeros((len(t_dt), m))

dt = 0.02 # time step in seconds

idx_time = t_dt <= 5
print('idx_time: ', idx_time)
u[:, 0] = 0.03
u[idx_time, 0] = 0.03 + 0.03 * np.sin(2*np.pi * 0.5 * t_dt[idx_time]**2 /5)

idx_time = t_dt <= 6
u[idx_time, 1] = 0.5
idx_time = (t_dt >= 6) & (t_dt <= 11)
u[idx_time, 1] = 0.02 + 0.02 * np.sin(2*np.pi * 0.5 * (t_dt[idx_time] - 6)**2 /5)
idx_time = t_dt >= 11
u[idx_time, 1] = 0.02

idx_time = t_dt <= 12
u[idx_time, 2] = 0.135
idx_time = (t_dt >= 12) & (t_dt <= 17)
u[idx_time, 2] = 0.02 + 0.02 * np.sin(2*np.pi * 0.5 * (t_dt[idx_time] - 12 )**2 /5)
idx_time = t_dt >= 17
u[idx_time, 2] = 0.02

plt.figure()
plt.plot(t_dt, u[:,0], '.', label='PQ_1 Input')
plt.plot(t_dt, u[:,1], '.', label='PQ_2 Input')
plt.plot(t_dt, u[:,2], '.', label='PQ_ Input')
plt.title('Inputs (pu)')
plt.xlabel('Time (s)')
plt.grid(True)
# plt.savefig('IEEE14_Inputs.png')

# Build DataFrame for easy handling
cols = ['t'] + [f'V_b{b}' for b in buses_of_interest]
print('time shape:', np.shape(time.reshape(-1,1)))
print('V_meas shape:', np.shape(V_meas))
# d = np.hstack([time.reshape(-1,1), V_meas])
d = np.hstack([time.reshape(-1, 1), V_meas])
data = data_cleaner.data_cleaner(d)

df = pd.DataFrame(data, columns=cols)
for j in range(u.shape[1]):
    df[f'u{j+1}'] = u[:,j]
print('Dataframe created with columns: \n', df)

plt.figure(figsize=(10,6))
for b in buses_of_interest:
    plt.plot(df['t'], df[f'V_b{b}'], label=f'Bus {b}')
plt.title('Selected bus voltages (pu)')
plt.xlabel('Time (s)')
plt.legend()
plt.grid(True)
# plt.savefig('IEEE14_Bus_Voltages.png')

# %%
# 3) Subspace identification (N4SID) to obtain a reduced-order discrete-time model
# We'll identify a single MIMO model with inputs u (3) and outputs y (voltages)
print("--------------------- 3. Performing Sys ID ---------------------")

Y = df[[f'V_b{b}' for b in buses_of_interest]].values
U = df[[f'u{j+1}' for j in range(u.shape[1])]].values

# Choose model order (experiment with this)
model_order = 20
print('Running N4SID...')
model = sysid(Y, U, 'N4SID', SS_fixed_order=model_order)
print('Finished identification.')

A_id = np.array(model.A)
B_id = np.array(model.B)
C_id = np.array(model.C)
D_id = np.array(model.D)

# Convert identified model to discrete-time state-space in control package
sys_id = ss(A_id, B_id, C_id, D_id)
# print('Identified discrete-time state-space model: \n', sys_id)
# %% 
# 4) Estimated Model Verification 
# Verify that the identified model is representative of the original system by simulating
# its response to the same inputs.
print("--------------------- 4. Verifying Identified Model ---------------------")

# First, identify initial state x0.
# To see if we can find the initial state, let's check the observability of the system. 
Observability_matrix = np.vstack([C_id @ np.linalg.matrix_power(A_id, i) for i in range(model_order)])
rank_of_observability = np.linalg.matrix_rank(Observability_matrix)
if rank_of_observability == model_order:
    print("The system is observable. Initial state can be determined from outputs.")
else:
    print("The system is NOT observable!!!!!!!!!!!!!!!!!!!!")

# x0_estimate = np.linalg.pinv(Observability_matrix) @ Y[0:model_order* C_id.shape[0], :].flatten()
ABC = np.hstack([Y[i, :] for i in range(model_order)])
# print("Shape of stacked outputs for initial state estimation: ", np.shape(ABC))
x0_estimate = np.linalg.pinv(Observability_matrix) @ ABC

print("Estimated initial state x0: ", x0_estimate)

# Controllability check 
Controllability_matrix = np.hstack([np.linalg.matrix_power(A_id, i) @ B_id for i in range(model_order)])
rank_of_controllability = np.linalg.matrix_rank(Controllability_matrix)
if rank_of_controllability == model_order:
    print("The system is controllable. Any state belonging in R^n can be reach in (at most) n steps.")
else:
    print("The system is NOT controllable!!!!!!!!!!!!!!!!!!!!")

# Now, simulate the identified model's response using the estimated initial state 
# and the original inputs u. 
y_id = np.zeros((len(t_dt), p))

x = x0_estimate
for i in range(len(t_dt)):
    u_short = u[i, :]
    y_id[i, :] = C_id @ x + D_id @ u_short
    x = A_id @ x + B_id @ u_short # update state
error = 100*np.abs(Y - y_id)/np.abs(Y)
print("Average Percent Error: ", np.mean(error))
print("Max Percent Error: ", np.max(error))

plt.figure()
plt.plot(df['t'], Y[:,0]   , color=[0     , 0.4470, 0.7410], linestyle='-' , label='ANDES Bus 2 Voltage')
plt.plot(df['t'], y_id[:,0], color=[0     , 0.4470, 0.7410], linestyle='--', label='Sys ID Bus 2 Voltage')
plt.plot(df['t'], Y[:,1]   , color=[0.8500, 0.3250, 0.0980], linestyle='-' , label='ANDES Bus 4 Voltage')
plt.plot(df['t'], y_id[:,1], color=[0.8500, 0.3250, 0.0980], linestyle='--', label='Sys ID Bus 4 Voltage')
plt.plot(df['t'], Y[:,2]   , color=[0.9290, 0.6940, 0.1250], linestyle='-' , label='ANDES Bus 7 Voltage')
plt.plot(df['t'], y_id[:,2], color=[0.9290, 0.6940, 0.1250], linestyle='--', label='ANDES Bus 7 Voltage')
plt.plot(df['t'], Y[:,3]   , color=[0.4940, 0.1840, 0.5560], linestyle='-' , label='ANDES Bus 9 Voltage')
plt.plot(df['t'], y_id[:,3], color=[0.4940, 0.1840, 0.5560], linestyle='--', label='ANDES Bus 9 Voltage')
plt.title('Outputs (pu)')
plt.xlabel('Time (s)')
plt.grid(True)
plt.legend()
# plt.savefig('IEEE14_Outputs_ANDES_vs_ID.png')

# %%
# 5) MPC formulation (discrete-time) using CVXPY
print("--------------------- 5. MPC Formulation ---------------------")

# General problem setup
T_sim = 10
dt_ctrl = dt * 5  # control every 5 simulator steps 
steps = int(T_sim/dt_ctrl)
print('Number of steps: ', steps)

N_horizon = 20    # prediction horizon

# Model parameters
n = model_order
u_0 = np.array([0.217, 0.5, 0.135]) # reference load (active power) input
# y_0 = np.array([1.0197, 0.99858, 1.00682, 1.00193])
y_0 = np.array([1.03, 1.01140345005209, 1.02247149945009, 1.02176878748932]) # reference bus voltages

# Variable declaration for logging
x_traj = np.zeros((n, steps + 1))
u_traj = np.zeros((m, steps))

# Initialize the system 
y_meas = y_0 
# approximate x0 from initial voltages and initialize x_traj 
x_0 = np.linalg.pinv(C_id) @ (y_0 - D_id @ u_0)
x_traj[:, 0] = x_0

# ASIDE: mini test 
y_current = C_id @ x_0 + D_id @ u_0
x_next = A_id @ x_0 + B_id @ u_0

print('current y: ', y_current)
print('error in y: ', np.abs(y_0 - y_current)/y_0*100)
print('current x: ', x_0[0:3])
print('next x: ', x_next[0:3])
print('avg error in x: ', np.mean(np.abs(x_0 - x_next)/x_0)*100)


# Optimization weights
Q = 1000*np.eye(p)
R = 10*np.eye(m)
S = 100*Q


time_cl = []
Vcl = []
Ucl = []

print('Starting closed-loop external control loop (MPC applied)...')
# case = 'IEEE_cases/ieee14_gentrip_v0.xlsx'
case = 'IEEE_cases/ieee14_base_v0.xlsx'
ss_cl = andes.load(case)

ss_cl.PQ.config.p2p = 1.0
ss_cl.PQ.config.p2i = 0
ss_cl.PQ.config.p2z = 0

ss_cl.PQ.config.q2q = 1.0
ss_cl.PQ.config.q2i = 0
ss_cl.PQ.config.q2z = 0

ss_cl.PQ.pq2z = 0
ss_cl.TDS.config.criteria = 0  # temporarily turn off stability criteria based on angle separation
ss_cl.TDS.config.tstep = dt_ctrl

ss_cl.PFlow.run()

# run ANDES forward by dt_ctrl
t = 0.0
i = 0

while t < T_sim - 1e-9:
# Reset ANDES to baseline and re-add disturbances (we will re-run dynamic sim in small increments)
# Define optimization variables for the current horizon
    x = cp.Variable((n, N_horizon + 1))
    u = cp.Variable((m, N_horizon))
    y = cp.Variable((p, N_horizon + 1))

    # Build cost function and constraints
    cost = 0
    constr = []

    ## Build the stage cost, system dynamics, and input constraints
    for k in range(N_horizon):
        # Stage cost
        cost += cp.quad_form(y[:, k] - y_0, Q) + cp.quad_form(u_0 - u[:, k], R)
        # System dynamics
        constr += [x[:, k + 1] == A_id @ x[:, k] + B_id @ u[:, k]]
        constr += [y[:, k]     == C_id @ x[:, k] + D_id @ u[:, k]]
        # Input constraints
        constr += [u[:, k] <= 5, u[:, k] >= 0]
    
    # Terminal constraints
    cost += cp.quad_form(y[:, k+1] - y_0, S)
    # Initial constraints
    constr += [x[:, 0] == x_traj[:, i]]   
    
    i += 1
    
    # Solve the finite-horizon optimal control problem
    problem = cp.Problem(cp.Minimize(cost), constr)
    problem.solve()

    # Apply first control input and update the system state
    u_opt = u[:, 0].value  
    print("optimal control input u at time ", t, " is ", u_opt)
    Ucl.append(u_opt)

    # Apply the control input by altering ANDES active power inputs
    ss_cl.PQ.Ppf.v[0] = u_opt[0]
    ss_cl.PQ.Ppf.v[1] = u_opt[1]
    ss_cl.PQ.Ppf.v[9] = u_opt[2]
    print('Active power fed into ANDES sim: ', ss_cl.PQ.Ppf.v)

    # run ANDES forward by dt_ctrl
    ss_cl.TDS.config.tf = t + dt_ctrl
    ss_cl.TDS.run()
    t = ss_cl.dae.ts.t[-1]
    print('ANDES stepped to t =', t)
    time_cl.append(t)

    # read ANDES bus voltage data/measurements
    Vbus_cl = ss_cl.dae.ts.y[:, ss_cl.Bus.v.a]
    Vmeas_cl = Vbus_cl[-1, [b-1 for b in buses_of_interest]]
    y_meas = np.hstack([Vmeas_cl])

    # Estimate state from inputs and outputs (yk = C xk + D uk)
    x_est = np.linalg.pinv(C_id) @ (y_meas - D_id @ u_opt)
    # Store estimate state for use in receding horizon
    x_traj[:, i] = x_est
    
    Vcl.append(Vmeas_cl)

print('Closed-loop run complete')

# Convert stored arrays to numpy for plotting
time_cl = np.array(time_cl)
Vcl = np.vstack(Vcl)
Ucl = np.vstack(Ucl)

# %%
# 7) Plot closed-loop results
print("--------------------- 7. Plotting Results ---------------------")
color = ['blue', 'orange', 'green', 'red']
plt.figure(figsize=(10,6))
for i,b in enumerate(buses_of_interest):
    plt.plot(time_cl, Vcl[:,i], label=f'Bus {b}', color = color[i])
    plt.axhline(y_0[i], color=color[i], linestyle='--')
plt.title('Closed-loop selected bus voltages (pu)')
plt.xlabel('Time (s)')
plt.legend()
plt.grid(True)
plt.savefig('fig2_closed_loop_voltages.png')

plt.figure(figsize=(8,4))
for i in range(Ucl.shape[1]):
    plt.step(time_cl, Ucl[:,i], where='post', label=f'u{i+1}')
plt.title('Applied MPC control inputs')
plt.xlabel('Time (s)')
plt.legend()
plt.grid(True)
plt.savefig('fig3_closed_loop_controls.png')

print('Notebook finished')


