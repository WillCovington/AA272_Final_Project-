import matplotlib.pyplot as plt

def plot_cn0_vs_time(sat_df, sat_label=None):
    t0 = sat_df["time"].min()
    t_rel = sat_df["time"] - t0

    fig, ax = plt.subplots()
    ax.plot(t_rel, sat_df["Cn0DbHz"])
    ax.set_xlabel("Time since start [s]")
    ax.set_ylabel("C/N0 [dB-Hz]")
    if sat_label:
        ax.set_title(f"CN0 vs Time – {sat_label}")
    ax.grid(True)
    return fig, ax