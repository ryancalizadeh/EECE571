import andes

# Load the IEEE 14-bus system shipped with ANDES
ss = andes.load("OLD_IEEE/ieee14_alter.xlsx")

# Example: modify the PQ load at Bus 4
# Let's increase the active power P from 0.478 to 0.6 pu at t = 1.0s
ss.PQ.config.p2p = 1.0
ss.PQ.config.p2i = 0
ss.PQ.config.p2z = 0

ss.PQ.config.q2q = 1.0
ss.PQ.config.q2i = 0
ss.PQ.config.q2z = 0

ss.PFlow.run()

# Run dynamic simulation until t = 10 seconds
ss.TDS.config.tf = 10
ss.TDS.config.criteria = 0  # temporarily turn off stability criteria based on angle separation
ss.TDS.run()

# print(ss.PQ.as_df())

# Get frequency, voltages, states, etc.
ss.TDS.load_plotter()

# fig, ax = ss.TDS.plt.plot((6, 7, 8, 9))
# fig.savefig("IEEE14_alter_plot.png", dpi=300)


fig, ax = ss.TDS.plt.plot(ss.GENROU.omega)
fig.savefig("IEEE14_alter_omega.png", dpi=300)

fig, ax = ss.TDS.plt.plot(ss.GENROU.v)
fig.savefig("IEEE14_alter_v.png", dpi=300)