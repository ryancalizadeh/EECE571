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
from control import ss, forced_response

from funcs import *

# %%
# 1) Load IEEE-14 and run an open-loop simulation to collect data
# The aim: excite both active and reactive dynamics with several disturbances
print("--------------------- 1. Running Dynamic Simulation ---------------------")

# case = 'kundur_full.xlsx'  
case = 'IEEE_cases/ieee14_alter_v5.xlsx'  
# case = 'IEEE_cases/ieee14_alter_v4.xlsx'  
# case = 'IEEE_cases/ieee14_alter_v3.xlsx'  
# case = 'IEEE_cases/ieee14_alter_v0.xlsx'  
# case = 'IEEE_cases/ieee14_v0.xlsx'  

T_END = 20.0


# ssys = ANDES_Case_Runner(case, T_END) 
ssys = pd.read_csv('ieee14_alter_v5_out.csv')

# %%
# 2) Extract time-series data
print("--------------------- 2. Extracting Time-Series Data ---------------------")

time = ssys['Time [s]'].values
n_t = len(time)
print('Time samples:', n_t)
t_dt = data_cleaner(time)

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
u[idx_time, 0] = 0.03 * np.sin(2*np.pi * 0.5 * t_dt[idx_time]**2 /5)

idx_time = (t_dt >= 6) & (t_dt <= 11)
u[idx_time, 1] = 0.02 * np.sin(2*np.pi * 0.5 * (t_dt[idx_time] - 6)**2 /5)

idx_time = (t_dt >= 12) & (t_dt <= 17)
u[idx_time, 2] = 0.015 * np.sin(2*np.pi * 0.5 * (t_dt[idx_time] - 12 )**2 /5)

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
d = np.hstack([time.reshape(-1, 1), V_meas])
data = data_cleaner(d)

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
# 5) MPC formulation (discrete-time) using CasADi
# We design an MPC to regulate the measured outputs (voltages) by manipulating
# the inputs. For this example, inputs represent fast-acting setpoint adjustments (synthetic).
print("--------------------- 5. MPC Formulation ---------------------")

# Create CasADi variables using the discrete identified model
n = model_order

# Convert numpy arrays to CasADi DM
A_ca = ca.DM(A_id)
B_ca = ca.DM(B_id)
C_ca = ca.DM(C_id)
D_ca = ca.DM(D_id)

# MPC parameters
N_horizon = 20  # prediction horizon
# dt_sample = time[1] - time[0]  # uniform sampling
print('Sampling dt from ANDES data:', dt)

# Define weighting matrices
Q = 100*np.diag([1]*len(buses_of_interest))  # penalize voltages 
print('Q Matrix: ', Q)
R = 10 * np.eye(m)

# Build MPC optimization problem
opti = ca.Opti()

# decision variables: sequences of future control moves (u0...u_{N-1}) and initial state
U_var = opti.variable(m, N_horizon)
X_var = opti.variable(n, N_horizon+1)

# parameter: initial state and reference
X0_param = opti.parameter(n)
Yref_param = opti.parameter(p, N_horizon)

# initial condition constraint
opti.subject_to(X_var[:,0] == X0_param)

# Initialize cost
cost = 0
# Initialize active power
P = opti.variable(m, N_horizon)
opti.subject_to(P[:,0] == np.array([0.217, 0.5, 0.135]))
# P = np.array([0.217, 0.5, 0.135])

for k in range(N_horizon):
    # predicted output y_k = C x_k + D u_k
    yk = C_ca @ X_var[:,k] + D_ca @ U_var[:,k]
    # reference at step k
    yrefk = Yref_param[:,k]
    print("yrefk: ", yrefk)
    # stage cost
    cost += ca.mtimes([(yk - yrefk).T, ca.DM(Q), (yk - yrefk)]) + ca.mtimes([U_var[:,k].T, ca.DM(R), U_var[:,k]])
    # dynamics constraint: x_{k+1} = A x_k + B u_k
    xnext = A_ca @ X_var[:,k] + B_ca @ U_var[:,k]
    opti.subject_to(X_var[:,k+1] == xnext)
    # P[:, k] = np.array([0.217, 0.5, 0.135]) if k==0 else P[:, k-1] + U_var[:, k]
    P[:, k] = P[:, k-1] + U_var[:, k]
    opti.subject_to(opti.bounded(0, P, 5))

    # opti.subject_to(P >= 0)
    # opti.subject_to(P <= 5)

opti.minimize(cost)

# input constraints (example saturations)
# P = opti.variable(m, N_horizon)
# P0 = np.array([
#     [0.217] * N_horizon,
#     [0.5]   * N_horizon,
#     [0.135] * N_horizon
# ])
# P = U_var + P0

# P_min = 0
# P_max = 3 
opti.subject_to(opti.bounded(0, P, 5))

# opti.subject_to(opti.bounded(u_min, U_var, u_max))

# solver settings
opts = {'print_time': False, 'ipopt': {'print_level': 0, 'max_iter':1000}}
opti.solver('ipopt', opts)

print('MPC problem built')

# %%
# 6) MPC closed-loop simulation with ANDES - external control loop
# We will run a closed-loop where at each control sample we:
#  - estimate reduced state x (we'll initialize with zeros and then use model + measurements)
#  - solve MPC for N_horizon and apply only the first control move to ANDES
#  - step ANDES forward by dt_ctrl and repeat
print("--------------------- 6. Running MPC with ANDES ---------------------")

# For state estimation we will use a simple approach: compute x by solving least squares
# from measured outputs y = C x + D u  -> x_est = pinv(C) (y - D u). This is simple but depends
# on C having full column rank or using an approximate estimator. For better performance use
# a Kalman filter or an observer designed from the identified model.

# Precompute pseudo-inverse for quick estimation
C_np = np.array(C_id)
D_np = np.array(D_id)
C_pinv = np.linalg.pinv(C_np)


# control sample time (match identified sampling or choose coarser for computational tractability)
dt_ctrl = dt * 5  # control every 5 simulator steps (adjust if needed)
print('Control sample time dt_ctrl:', dt_ctrl)

t = 0.0
t_end = 10

# storage for closed-loop signals
time_cl = []
Vcl = []
Ucl = []
Pcl = []

# initialize state estimate
x_est = np.zeros(n)

# run loop
print('Starting closed-loop external control loop (MPC applied)...')
case = 'IEEE_cases/ieee14_gentrip_v0.xlsx'
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
i = 0
while t < t_end - 1e-9:
    # run ANDES forward by dt_ctrl
    print('Active power: ', ss_cl.PQ.Ppf.v)
    ss_cl.TDS.config.tf = t + dt_ctrl
    ss_cl.TDS.run()
    t = ss_cl.dae.ts.t[-1]
    print('ANDES stepped to t =', t)
    time_cl.append(t)

    # read measurements
    ts_cl = ss_cl.dae.ts
    Vbus_cl = ss_cl.dae.ts.y[:, ss_cl.Bus.v.a]
    Vmeas_cl = Vbus_cl[-1, [b-1 for b in buses_of_interest]]
    y_meas = np.hstack([Vmeas_cl])

    # reconstruct approximate state from y = C x + D u  => x = pinv(C) (y - D u)
    # use last applied u (if none applied yet, assume zero)
    last_u = np.zeros(m) if len(Ucl)==0 else Ucl[-1] # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    x_est = C_pinv @ (y_meas - D_np @ last_u)

    # build reference trajectory Yref: zeros (aim for nominal voltages)
    # here y references: voltages -> 1.0 pu
    y_ref_step = np.hstack([1.0197, 0.99858, 1.00682, 1.00193]) # nominal voltages from initial sim
    Yref = np.tile(y_ref_step.reshape(-1,1), (1, N_horizon))

    # set MPC parameters and initial condition
    opti.set_value(X0_param, x_est)
    opti.set_value(Yref_param, Yref)

    # initial guess (warm start) optional
    opti.set_initial(X_var, np.tile(x_est.reshape(-1,1), (1, N_horizon+1)))
    opti.set_initial(U_var, np.zeros((m, N_horizon)))

    try:
        sol = opti.solve()
        u_opt = np.array(sol.value(U_var)[:,0]).flatten()
        p_opt = np.array(sol.value(P)[:, 0]).flatten()
        print('P: ', p_opt)
    except Exception as e:
        print('MPC solve failed at t=', t, 'error:', e)
        break
        # fallback: zero control
        u_opt = np.zeros(m)
    print('previous active Power Ppf values: ', [ss_cl.PQ.Ppf.v[0],ss_cl.PQ.Ppf.v[1], ss_cl.PQ.Ppf.v[9]])
    ss_cl.PQ.Ppf.v[0] = ss_cl.PQ.Ppf.v[0] + u_opt[0]
    ss_cl.PQ.Ppf.v[9] = ss_cl.PQ.Ppf.v[1] + u_opt[2]
    ss_cl.PQ.Ppf.v[1] = ss_cl.PQ.Ppf.v[2] + u_opt[1]
    print('new active power Ppf values: ', [ss_cl.PQ.Ppf.v[0],ss_cl.PQ.Ppf.v[1], ss_cl.PQ.Ppf.v[9]])

    # apply u_opt to ANDES: for this notebook the inputs represent scheduled active power injections
    # We'll emulate this by adding a custom injection action or by modifying load offsets.
    # As a simple approach, modify a set of pseudo 'battery' injections at certain buses maintained
    # as time-series events. If ANDES does not support direct runtime injection updates, you may
    # instead update generator reference setpoints or exciter inputs. Adjust this to your environment.
    print("optimal control input u at time ", t, " is ", u_opt)
    # For demonstration, store the control and continue (user should map these to actual model inputs)
    Pcl.append(p_opt)
    Ucl.append(u_opt)
    Vcl.append(Vmeas_cl)

    i += 1

    # In a real run you would call an API like ss_cl.set('Battery', dict(...)) or update generator setpoints.
    # Example placeholder: ss_cl.set_input_vector(u_opt)

print('Closed-loop run complete')

# Convert stored arrays to numpy for plotting
time_cl = np.array(time_cl)
Vcl = np.vstack(Vcl)
Ucl = np.vstack(Ucl)
Pcl = np.vstack(Pcl)
# %%
# 7) Plot closed-loop results
print("--------------------- 7. Plotting Results ---------------------")
color = ['blue', 'orange', 'green', 'red']
plt.figure(figsize=(10,6))
for i,b in enumerate(buses_of_interest):
    plt.plot(time_cl, Vcl[:,i], label=f'Bus {b}', color = color[i])
    plt.axhline(Yref[i, 0], color=color[i], linestyle='--')
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

plt.figure(figsize=(8,4))
for i in range(Ucl.shape[1]):
    plt.step(time_cl, Pcl[:,i], where='post', label=f'P{i+1}', linestyle='--')
plt.title('Active Power')
plt.xlabel('Time (s)')
plt.legend()
plt.grid(True)
plt.savefig('fig3_closed_loop_POWER.png')

print('Notebook finished')


