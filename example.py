import matplotlib
matplotlib.use('Qt5Agg')

import andes
import matplotlib.pyplot as plt
#from andes.utils.paths import get_case

if __name__ == '__main__':
    andes.config_logger(stream_level=50)
    ss = andes.run('kundur_full.xlsx', default_config=True, pert='pert.py')
    ss.TDS.config.diagnose = 1
    ss.TDS.config.verbose = 2
    ss.TDS.config.tf = 8
    ss.TDS.config.criteria = 0
    ss.TDS.config.tstep = 0.01

    for field in dir(ss.TDS.config):
        print(f'{field}: {getattr(ss.TDS.config, field)}')
    # exit()

    ss.TDS.run()
    print(ss.exit_code)
    if ss.TDS.load_plotter() is None:
        ss.TDS.load_plotter()

    fig, ax = ss.TDS.plt.plot((5, 6, 7, 8))
    fig.savefig("plot_w.png", dpi=300)

    fig, ax = ss.TDS.plt.plot((9, 10, 11, 12))
    fig.savefig("plot_e.png", dpi=300)