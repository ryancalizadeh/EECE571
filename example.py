import matplotlib
matplotlib.use('Qt5Agg')

import andes
import matplotlib.pyplot as plt
#from andes.utils.paths import get_case

if __name__ == '__main__':
    andes.config_logger(stream_level=50)
    ss = andes.run('kundur_full.xlsx', default_config=True)
    ss.TDS.config.tf = 20
    ss.TDS.run()
    print(ss.exit_code)
    if ss.TDS.load_plotter() is None:
        ss.TDS.load_plotter()

    fig, ax = ss.TDS.plt.plot((5, 6, 7, 8))
    fig.savefig("plot_w.png", dpi=300)

    fig, ax = ss.TDS.plt.plot((9, 10, 11, 12))
    fig.savefig("plot_e.png", dpi=300)