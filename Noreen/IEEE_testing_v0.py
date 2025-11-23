# %%
"""
Runnable Jupyter-style notebook (Python) for:
Centralized MPC on the IEEE 14-bus system using ANDES.

Notes:
- The notebook uses cell markers (# %% ) so you can drop it into a .py and open as a notebook,
  or copy cells into a Jupyter notebook directly.
- It performs the following high-level steps:
  1. Install & import required packages (andes, sippy, control, casadi, numpy, pandas, matplotlib)
  2. Load IEEE-14 case from ANDES and run baseline simulation with disturbances
  3. Collect time-series data (bus voltages, generator speeds, inputs)
  4. Identify a reduced-order state-space model (N4SID via sippy)
  5. Formulate and test a centralized MPC (CasADi) on the identified discrete model
  6. Implement an external discrete-time control loop that runs ANDES and applies MPC actions
  7. Validate closed-loop performance and plot results

Assumptions & caveats:
- This notebook assumes ANDES is available via `pip install andes-dyn` or similar. If your
  ANDES installation differs (e.g. local install), adjust the import/load steps accordingly.
- The notebook uses sippy for subspace ID and casadi for MPC. If any package is missing,
  install via pip in the cell below.
- The identified model is a reduced-order linear discrete-time system. The MPC is designed
  on that model and applied in closed-loop to the full nonlinear ANDES simulator via an
  external sample-and-hold control loop.

If anything fails due to package versions or environment, you can still use this as a
clear blueprint to adapt to your local setup.
"""

# %%
# 0) Install required packages (run this cell first if running in a fresh environment)
# NOTE: comment out or modify installs if packages are already present or if you use conda.
 
import sys
import subprocess
import pkgutil

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
# 1) Imports
import numpy as np
import pandas as pd
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

# %%
# 2) Load IEEE-14 and run an open-loop simulation to collect data
# The aim: excite both active and reactive dynamics with several disturbances

case = 'kundur_full.xlsx'  
# case = 'IEEE14Bus/ieee14.xlsx'  

ssys = andes.load(case)

# Small helper to add step disturbances at given bus
def add_load_step(sim, bus, dP, dQ, t):
    # adapt to ANDES API - using the same pattern used earlier
    sim.add('Load', dict(bus=bus, id=f'STEP_b{bus}_t{t}', dP=dP, dQ=dQ, t=t))

# # Add multiple load steps at different times/locations to create richer data
# add_load_step(ssys, bus=2, dP=0.05, dQ=0.0, t=1.0)
# add_load_step(ssys, bus=6, dP=-0.04, dQ=0.0, t=4.0)
# add_load_step(ssys, bus=9, dP=0.03, dQ=0.0, t=8.0)

# Run power flow and dynamic sim
# ssys.setup()
ssys.PFlow.run()
T_END = 20.0
ssys.TDS.config.tf = T_END
print('Running ANDES dynamic simulation...')
ssys.TDS.run()
# res = ssys.run(t_end=T_END)
print('Simulation finished')

ssys.TDS.load_plotter()

fig, ax = ssys.TDS.plt.plot((6, 7, 8))
fig.savefig("IEEE14_plot.png", dpi=300)
# %%
# 3) Extract time-series data
# The exact ts key names may depend on ANDES distribution; adapt below if keys differ.

ts = ssys.dae.ts
# print(dir(ts))
# Common arrays
time = ts.t
# print("time: 'n", time)
# Generator rotor speeds (omega) and bus voltages
# omega = np.array(ts['Syn.omega'])    # shape (n_t, n_gens)
omega = ssys.dae.ts.x[:, ssys.GENROU.omega.a]
# print("omega: \n", omega)
# Vbus = np.array(ts['Bus.V'])         # shape (n_t, n_buses)
Vbus = ssys.dae.ts.y[:, ssys.Bus.v.a]
print("Vbus: \n", Vbus)

n_t = len(time)
print('Time samples:', n_t)

# Choose outputs and inputs for identification & MPC
# Outputs: system-wide frequency estimate and voltages at selected buses
# For frequency we use average generator speed deviation (omega - 1.0 pu)

omega_mean = np.mean(omega, axis=1)  # average rotor speed across synchronous machines
freq_dev = omega_mean - 1.0          # per-unit deviation from nominal

# Select voltage measurements at a subset of buses (e.g., buses [2,4,7,9,14])
buses_of_interest = [2,4,7,9]
V_meas = Vbus[:, [b-1 for b in buses_of_interest]]  # ANDES buses may be 1-indexed

# Inputs: construct a synthetic input vector representing the known load steps
# We'll approximate the load steps by reconstructing them from ANDES events: here we know
# dP at times 1,4,8 for buses 2,6,9 respectively
u = np.zeros((n_t, 3))
for i,t in enumerate(time):
    u[i,0] = 0.05 if t >= 1.0 else 0.0
    u[i,1] = -0.04 if t >= 4.0 else 0.0
    u[i,2] = 0.03 if t >= 8.0 else 0.0

# Build DataFrame for easy handling
cols = ['t','freq_dev'] + [f'V_b{b}' for b in buses_of_interest]
data = np.hstack([time.reshape(-1,1), freq_dev.reshape(-1,1), V_meas])

df = pd.DataFrame(data, columns=cols)
print('Dataframe created with columns: \n', df)
# add inputs
for j in range(u.shape[1]):
    df[f'u{j+1}'] = u[:,j]

print('Dataframe shape:', df.shape)

# Quick plots of signals
plt.figure()
plt.show()
plt.plot(df['t'], df['freq_dev'])
plt.title('Average frequency deviation (pu)')
plt.xlabel('Time (s)')
plt.grid(True)

plt.figure(figsize=(10,6))
for b in buses_of_interest:
    plt.plot(df['t'], df[f'V_b{b}'], label=f'Bus {b}')
plt.title('Selected bus voltages (pu)')
plt.xlabel('Time (s)')
plt.legend()
plt.grid(True)
plt.savefig('my_plot.png')
# %%
# 4) Subspace identification (N4SID) to obtain a reduced-order discrete-time model
# We'll identify a single MIMO model with inputs u (3) and outputs y (freq + voltages)

Y = df[['freq_dev'] + [f'V_b{b}' for b in buses_of_interest]].values
U = df[[f'u{j+1}' for j in range(u.shape[1])]].values

# Downsample data to reduce computation for identification (optional)
# Here we assume ANDES ts is already reasonably coarse (e.g., 0.01-0.02s). If it's too fine,
# consider resampling. For now, use the raw data.

# Choose model order (experiment with this)
model_order = 8
print('Running N4SID (this may take a few seconds)...')
model = sysid(Y, U, 'N4SID', SS_fixed_order=model_order)

A_id = np.array(model.A)
B_id = np.array(model.B)
C_id = np.array(model.C)
D_id = np.array(model.D)

print('Identified model shapes: A', A_id.shape, 'B', B_id.shape, 'C', C_id.shape, 'D', D_id.shape)

# Convert identified model to discrete-time state-space in control package
sys_id = ss(A_id, B_id, C_id, D_id)
print('Identified discrete-time state-space model: \n', sys_id)
# %%
# 5) MPC formulation (discrete-time) using CasADi
# We design an MPC to regulate the measured outputs (frequency dev + voltages) by manipulating
# the inputs. For this example, inputs represent fast-acting setpoint adjustments (synthetic).

# Create CasADi variables using the discrete identified model
n = A_id.shape[0]
m = B_id.shape[1]
p = C_id.shape[0]

# Convert numpy arrays to CasADi DM
A_ca = ca.DM(A_id)
B_ca = ca.DM(B_id)
C_ca = ca.DM(C_id)
D_ca = ca.DM(D_id)

# MPC parameters
N_horizon = 20  # prediction horizon
dt_sample = time[1] - time[0]  # uniform sampling
print('Sampling dt from ANDES data:', dt_sample)

# Define weighting matrices
Qy = np.diag([100.0] + [50.0]*len(buses_of_interest))  # penalize freq dev heavily and voltages moderately
Ru = 0.1 * np.eye(m)

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

# cost
cost = 0
for k in range(N_horizon):
    # predicted output y_k = C x_k + D u_k
    yk = C_ca @ X_var[:,k] + D_ca @ U_var[:,k]
    # reference at step k
    yrefk = Yref_param[:,k]
    # stage cost
    cost += ca.mtimes([(yk - yrefk).T, ca.DM(Qy), (yk - yrefk)]) + ca.mtimes([U_var[:,k].T, ca.DM(Ru), U_var[:,k]])
    # dynamics constraint: x_{k+1} = A x_k + B u_k
    xnext = A_ca @ X_var[:,k] + B_ca @ U_var[:,k]
    opti.subject_to(X_var[:,k+1] == xnext)

opti.minimize(cost)

# input constraints (example saturations)
u_min = -0.2
u_max = 0.2
opti.subject_to(opti.bounded(u_min, U_var, u_max))

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

# For state estimation we will use a simple approach: compute x by solving least squares
# from measured outputs y = C x + D u  -> x_est = pinv(C) (y - D u). This is simple but depends
# on C having full column rank or using an approximate estimator. For better performance use
# a Kalman filter or an observer designed from the identified model.

# Precompute pseudo-inverse for quick estimation
C_np = np.array(C_id)
D_np = np.array(D_id)
C_pinv = np.linalg.pinv(C_np)

# Reset ANDES to baseline and re-add disturbances (we will re-run dynamic sim in small increments)
ss_cl = andes.load(case)
# add_load_step(ss_cl, bus=2, dP=0.05, dQ=0.0, t=1.0)
# add_load_step(ss_cl, bus=6, dP=-0.04, dQ=0.0, t=4.0)
# add_load_step(ss_cl, bus=9, dP=0.03, dQ=0.0, t=8.0)
ss_cl.PFlow.run()

# control sample time (match identified sampling or choose coarser for computational tractability)
dt_ctrl = dt_sample * 5  # control every 5 simulator steps (adjust if needed)
print('Control sample time dt_ctrl:', dt_ctrl)

t = 0.0
t_end = T_END

# storage for closed-loop signals
time_cl = []
freq_cl = []
Vcl = []
Ucl = []

# initialize state estimate
x_est = np.zeros(n)

# run loop
print('Starting closed-loop external control loop (MPC applied)...')
while t < t_end - 1e-9:
    # run ANDES forward by dt_ctrl
    ss_cl.TDS.run(tf=t + dt_ctrl)
    t = ss_cl.dae.ts.t[-1]
    time_cl.append(t)

    # read measurements
    ts_cl = ss_cl.dae.ts
    omega_cl = ss_cl.dae.ts.x[:, ss_cl.GENROU.omega.a]
    Vbus_cl = ss_cl.dae.ts.y[:, ss_cl.Bus.v.a]
    omega_mean_cl = np.mean(omega_cl, axis=1)[-1]
    freq_dev_cl = omega_mean_cl - 1.0
    Vmeas_cl = Vbus_cl[-1, [b-1 for b in buses_of_interest]]
    y_meas = np.hstack([freq_dev_cl, Vmeas_cl])

    # reconstruct approximate state from y = C x + D u  => x = pinv(C) (y - D u)
    # use last applied u (if none applied yet, assume zero)
    last_u = np.zeros(m) if len(Ucl)==0 else Ucl[-1]
    x_est = C_pinv @ (y_meas - D_np @ last_u)

    # build reference trajectory Yref: zeros (aim for zero freq dev and nominal voltages 1.0)
    # here y references: freq_dev -> 0, voltages -> 1.0 pu
    y_ref_step = np.hstack([0.0, np.ones(len(buses_of_interest))])
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
    except Exception as e:
        print('MPC solve failed at t=', t, 'error:', e)
        # fallback: zero control
        u_opt = np.zeros(m)

    # apply u_opt to ANDES: for this notebook the inputs represent scheduled active power injections
    # We'll emulate this by adding a custom injection action or by modifying load offsets.
    # As a simple approach, modify a set of pseudo 'battery' injections at certain buses maintained
    # as time-series events. If ANDES does not support direct runtime injection updates, you may
    # instead update generator reference setpoints or exciter inputs. Adjust this to your environment.

    # For demonstration, store the control and continue (user should map these to actual model inputs)
    Ucl.append(u_opt)
    freq_cl.append(freq_dev_cl)
    Vcl.append(Vmeas_cl)

    # In a real run you would call an API like ss_cl.set('Battery', dict(...)) or update generator setpoints.
    # Example placeholder: ss_cl.set_input_vector(u_opt)

print('Closed-loop run complete')

# Convert stored arrays to numpy for plotting
time_cl = np.array(time_cl)
freq_cl = np.array(freq_cl)
Vcl = np.vstack(Vcl)
Ucl = np.vstack(Ucl)

# %%
# 7) Plot closed-loop results
plt.figure()
plt.plot(time_cl, freq_cl)
plt.title('Closed-loop average frequency deviation (pu)')
plt.xlabel('Time (s)')
plt.grid(True)
plt.savefig('fig1_closed_loop_freq.png')

plt.figure(figsize=(10,6))
for i,b in enumerate(buses_of_interest):
    plt.plot(time_cl, Vcl[:,i], label=f'Bus {b}')
plt.title('Closed-loop selected bus voltages (pu)')
plt.xlabel('Time (s)')
plt.legend()
plt.grid(True)
plt.savefig('fig2_closed_loop_voltages.png')

plt.figure(figsize=(8,4))
for i in range(Ucl.shape[1]):
    plt.step(time_cl, Ucl[:,i], where='post', label=f'u{i+1}')
plt.title('Applied MPC control moves (first applied move each step)')
plt.xlabel('Time (s)')
plt.legend()
plt.grid(True)
plt.savefig('fig3_closed_loop_controls.png')

print('Notebook finished')
