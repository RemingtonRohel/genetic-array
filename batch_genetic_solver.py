import numpy as np
import argparse
import matplotlib.pyplot as plt
import h5py

from genetic_solver import GeneticSolver
from plotting import Multiple
import array_factor


def plot_trials(weights, gains, angles, title, filename):
    """Plot the gain pattern in the horizontal plane and phases of each antenna.

    Parameters
    ----------
    weights: list of np.array
        List of arrays containing complex weights for each antenna.
    gains: list
        List of arrays with gain values. Array dimensions should match 'angles'
    angles: np.array
        Array of angular directions corresponding to gain patterns.
    title: str
        Title of the plot.
    filename: str
        Name of file to save plot as.
    """
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
    fig.suptitle(title)

    for i in range(len(gains)):
        ax1.plot(angles, gains[i])

    ax1.axvline(x=-24.3, color='k')
    ax1.axvline(x=24.3, color='k')
    ax1.yaxis.grid()
    ax1.set_xlabel('Azimuth (degrees)')
    ax1.set_ylabel('Gain (dB)')

    for i in range(len(weights)):
        ax2.plot(range(weights[i].size), np.unwrap(np.angle(weights[i])))

    major = Multiple(denominator=1)
    minor = Multiple(denominator=4)
    ax2.yaxis.set_major_locator(major.locator())
    ax2.yaxis.set_minor_locator(minor.locator())
    ax2.yaxis.set_major_formatter(major.formatter())
    ax2.yaxis.grid()

    ax2.set_xlabel("Antenna Index")
    ax2.set_ylabel("Phase (radians)")
    plt.savefig(filename, bbox_inches='tight')
    plt.close()


def main():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--resolution', type=float, help='angular resolution in degrees', default=0.5)
    parser.add_argument('--spacing', type=float, help='antenna spacing in meters', default=15.24)
    parser.add_argument('--left-bound', type=float, help='left bound of FOV in degrees CW from boresight', default=-27.0)
    parser.add_argument('--right-bound', type=float, help='right bound of FOV in degrees CW from boresight', default=27.0)
    parser.add_argument('outdir', type=str, help='directory to put results and plots in')
    parser.add_argument('freqs', nargs='+', type=float, help='frequencies in kHz to optimize for')
    args = parser.parse_args()

    num_antennas = 16   # main array
    angular_res = args.resolution  # degrees
    left_bound = args.left_bound  # FOV boundary, in degrees right of boresight
    right_bound = args.right_bound  # FOV boundary, in degrees right of boresight
    elevation = 0       # degrees up from horizon
    antenna_spacing = args.spacing     # meters
    labels = []

    # Position of each antenna along y-axis
    antenna_positions = array_factor.default_antenna_positions(num_antennas, antenna_spacing)

    # Arrays of spherical coordinate points
    els = np.array([elevation])
    azimuths = np.arange(-90, 90, angular_res)
    passband = (left_bound, right_bound)

    def add_configuration(w, label):
        """Add a set of weights with associated label for simulation."""
        weights.append(w)
        labels.append(label)

    # Common mode frequencies
    freqs = [x * 1000 for x in args.freqs] # Hz  [10.4e6, 10.5e6, 10.6e6, 10.7e6, 10.8e6, 10.9e6, 12.2e6, 12.3e6, 12.5e6, 13.0e6, 13.1e6, 13.2e6]  # Hz
    ripples = [2.0, 3.0, 4.0, 5.0]            # dB
    sidelobe_levels = [-12]                   # dB
    transition_widths = [5.0]                   # degrees
    population_size = 200
    azimuthal_points = 200
    num_trials = 10

    print("Ripples: {}".format(ripples))
    print("Sidelobe Levels: {}".format(sidelobe_levels))
    print("Transition Widths: {}".format(transition_widths))
    print("Freqs: {}".format(freqs))

    for freq in freqs:
        print("Frequency: {:.3f} MHz".format(freq * 1e-6))
        best_weights = []
        best_scores = []
        perfect = False
        for ripple in ripples:
            if perfect:
                break
            for sidelobe in sidelobe_levels:
                for width in transition_widths:
                    weights = []
                    scores = []
                    for trial in range(num_trials):
                        gs = GeneticSolver(num_antennas, antenna_spacing, freq, ripple, sidelobe, passband, width,
                                           population_size, azimuthal_points)
                        add_configuration(gs.weights, 'Trial {}'.format(trial))
                        scores.append(gs.best_score)
                        if gs.best_score == 0.0:
                            break

                    weights = np.array(weights)
                    scores = np.array(scores)

                    best_score_idx = np.argsort(scores)[0]
                    best_scores.append(scores[best_score_idx])
                    best_weights.append(weights[best_score_idx, :])

                    perfect = (scores[best_score_idx] == 0.0)

                    # Compute the array factor for each set of weights.
                    af = array_factor.array_factor(weights, antenna_positions, freq, els, azimuths)

                    gains = [af[i, 0, :] for i in range(af.shape[0])]

                    # Plot the results for this configuration
                    title = '{:.3f} MHz, {:.1f}dB Ripple, {:.1f}dB Peak Sidelobe, {:.1f} degree Transition Width, {} Antennas' \
                            ''.format(freq * 1e-6, ripple, sidelobe, width, num_antennas)
                    filename = f'{args.outdir}/Genetic_Solutions_{freq*1e-6:.3f}MHz_{ripple:.1f}ripple_{sidelobe:.1f}sidelobe_{width:.1f}width_{num_antennas}antennas.png'
                    plot_trials(weights, gains, azimuths, title, filename)

        best_weights = np.array(best_weights)
        best_scores = np.array(best_scores)

        best_weights_deg = np.rad2deg(np.unwrap(np.angle(best_weights), axis=-1))

        for s, w in zip(best_scores, best_weights):
            variable_phases = np.rad2deg(np.unwrap(np.angle(w)))
            print(f"Score: {s:.3f}\tPhases (deg): {variable_phases}")

        # Compute the array factor for each set of weights.
        af = array_factor.array_factor(best_weights, antenna_positions, freq, els, azimuths)

        gains = [af[i, 0, :] for i in range(af.shape[0])]

        outfile = f'{args.outdir}/results.hdf5'
        with h5py.File(outfile, 'a') as f:
            group_name = f'{int(freq/1000):05d}'
            g = f.create_group(group_name)
            g.create_dataset('phases', data=best_weights_deg)
            g.create_dataset('scores', data=best_scores)
            g.create_dataset('array_factor', data=np.array(gains))
            g.create_dataset('grid', data=azimuths)

        # Plot the results for this configuration
        title = '{:.3f} MHz'.format(freq * 1e-6)
        filename = '{}/Genetic_Solutions_{:.1f}MHz_{}antennas.png' \
                   ''.format(args.outdir, freq * 1e-6, num_antennas)
        plot_trials(best_weights, gains, azimuths, title, filename)


if __name__ == '__main__':
    main()

