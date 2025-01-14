import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import scipy

import array_factor as af


def cached_weights(num_antennas, freq):
    # {frequency in Hz : phases for the first half of the antennas, in degrees }
    cached_values_16_antennas = {
        12.0e6: [0., 164.32834753, 287.84658829, 347.83174452, 433.84544554, 532.75801346, 621.67692216, 594.45230541],
    }
    if num_antennas == 16:
        if freq in cached_values_16_antennas.keys():
            angles = cached_values_16_antennas[freq]
            phases = np.concatenate((angles, np.flip(angles)))
            weights = np.exp(-1j * np.deg2rad(phases))
        else:
            raise KeyError('Frequency not supported')
    else:
        raise ValueError('Number of antennas not supported')
    return weights


def calculate_directivities(weights, normalize=False):
    """Calculates the directivity for a set of weights"""
    # Position of each antenna along y-axis
    antenna_positions = np.arange(len(weights[0])) * antenna_spacing

    # Arrays of spherical coordinate points
    elevation = 0  # degrees up from horizon
    els = np.array([elevation])
    azimuths = np.arange(-90, 90, angular_res)

    # Compute the array factor for each set of weights.
    results = af.array_factor(weights, antenna_positions, freq, els, azimuths)

    if normalize:
        directivities = [results[i, 0, :] - np.max(np.abs(results[i, 0, :])) for i in range(results.shape[0])]
    else:
        directivities = [results[i, 0, :] for i in range(results.shape[0])]

    return directivities, azimuths, els


if __name__ == '__main__':
    plot_dir = '../../figures/wb'
    num_antennas = 16  # main array
    freq = 12.0e6  # Hz
    angular_res = 0.01  # degrees
    left_bound = -38.88  # FOV boundary, in degrees right of boresight
    right_bound = 38.88  # FOV boundary, in degrees right of boresight
    direction = 0.0  # degrees right of boresight
    antenna_spacing = 12.8016  # meters

    passband = (left_bound - direction, right_bound - direction)

    directions = np.linspace(-45, 45.01, 901)

    def add_configuration(w, l, c):
        """Add a set of weights with associated label for simulation."""
        weights.append(w)
        labels.append(l)
        colors.append(c)

    weights = []
    labels = []
    colors = []

    ##### Create the windowed directivity #####
    hamming_40 = np.array([0.08081232549588463, 0.12098514265395757, 0.23455777475180511, 0.4018918165398586,
              0.594054435182454, 0.7778186328978896, 0.9214100134552521, 1.0,
              1.0, 0.9214100134552521, 0.7778186328978896, 0.594054435182454,
              0.4018918165398586, 0.23455777475180511, 0.12098514265395757, 0.08081232549588463])
    chebyshev_30 = np.array([0.2910, 0.3173, 0.4557, 0.6018, 0.7424, 0.8637, 0.9528, 1.0000,
                             1.0000, 0.9528, 0.8637, 0.7424, 0.6018, 0.4557, 0.3173, 0.2910])

    # window = hamming_40
    window = chebyshev_30

    ### Load in the antenna factor from the NEC simulation ###
    nec_data = np.load("wallops/wallops_12000khz.npz")
    element_factor = nec_data["data"]
    azs = nec_data["az"]
    colat = nec_data["el"]
    colat_idx = 70
    interp_data = np.interp(np.arange(-90, 90, angular_res), azs - 90, element_factor[colat_idx, :])
    interp_data -= interp_data.max()

    intf_window = [0.1038961039, 1.0, 1.0, 0.1038961039]
    pointing_diffs = []
    for direction in directions:
        dirs, _, _ = calculate_directivities(
            [
                cached_weights(16, freq),
                window * af.linear_phase(af.default_antenna_positions(num_antennas, antenna_spacing=antenna_spacing), freq, direction)
            ],
            normalize=True,
        )
        intf_dirs, azs, _ = calculate_directivities([af.linear_phase(af.default_antenna_positions(4, antenna_spacing=antenna_spacing), freq, direction)],
                                                    normalize=True)
        dirs.append(intf_dirs[0])
        dirs = [np.power(10, d / 20) for d in dirs]
        dirs = [d * np.power(10, interp_data / 20) for d in dirs]  # add in element factor
        dirs = [d / np.max(np.abs(d)) for d in dirs]  # normalize by the peak value
        peak_dir = np.max(np.abs(dirs))

        multiplied = [dirs[i] * np.abs(dirs[0]) for i in range(1, len(dirs))]
        convolved = [multiplied[0] * np.conj(multiplied[0]),
                     multiplied[1] * np.conj(multiplied[1]),
                     multiplied[0] * np.conj(multiplied[1])]

        peaks = []
        widths = []
        medians = []
        height = -50
        for c in convolved:
            peak, props = scipy.signal.find_peaks(
                10 * np.log10(np.abs(c)),
                height=height,
                prominence=(5, None)
            )
            if len(peak) > 1:
                # Only keep the main lobe peak (should be the tallest peak)
                main_idx = np.argmax(props['peak_heights'])

                # Constrain width calculation to stop at the closest null between lobes
                prominences = props['prominences']
                left_bases = props['left_bases']
                right_bases = props['right_bases']

                prominence = prominences[main_idx]
                bounded = True
                if main_idx != 0:
                    left_base = left_bases[main_idx - 1]
                    if prominences[main_idx - 1] < prominence:
                        prominence = prominences[main_idx - 1]
                else:
                    left_base = left_bases[main_idx]
                    bounded = False
                if main_idx != len(peak) - 1:
                    right_base = right_bases[main_idx + 1]
                    if prominences[main_idx + 1] < prominence:
                        prominence = prominences[main_idx + 1]
                else:
                    right_base = right_bases[main_idx]
                    bounded = False

                peak = np.ones((1,), dtype=np.int64) * peak[main_idx]
                width = scipy.signal.peak_widths(
                    10 * np.log10(np.abs(c)),
                    peak,
                    prominence_data=(
                        np.ones((1,)) * prominence,
                        np.ones((1,), dtype=int) * left_base,
                        np.ones((1,), dtype=int) * right_base
                    ),
                    rel_height=1.0
                )
                widths.append(width)
                lower_bound = int(np.ceil(width[2])[0])
                upper_bound = int(np.floor(width[3])[0])
            else:
                continue
            peaks.append(peak)

            # Calculate mean of the main lobe
            # numerator = scipy.integrate.simpson(np.abs(c[lower_bound:upper_bound]) * azs[lower_bound:upper_bound], x=azs[lower_bound:upper_bound])
            # denominator = scipy.integrate.simpson(np.abs(c[lower_bound:upper_bound]), x=azs[lower_bound:upper_bound])
            # medians.append(numerator/denominator)

            # Calculate max of the main lobe
            # idx = np.argmax(np.abs(c[lower_bound:upper_bound]))
            # medians.append(azs[lower_bound:upper_bound][idx])

            # Calculate median of the main lobe
            cumsum = np.cumsum(np.abs(c[lower_bound:upper_bound]))
            cdf = cumsum / cumsum[-1]
            medians.append(azs[lower_bound:upper_bound][np.argwhere(cdf >= 0.5)[0]][0])

        pointing_diffs.append(medians)

        if np.abs(direction + 40.0) < 0.02:
            fig, ax = plt.subplots(1, 1, figsize=(6, 4))
            ax.yaxis.grid(visible=True)
            fig.tight_layout()
            labels = ['Main', 'Intf', 'Cross']
            colors = ['tab:blue', 'tab:orange', 'tab:green']
            af.plot_horizontal_gain(fig, ax, 20 * np.log10(np.abs([dirs[0]])), azs, ['$D_t$'], ['tab:purple'], bounds=False)
            af.plot_horizontal_gain(fig, ax, 10 * np.log10(np.abs(convolved)), azs, labels, colors, bounds=False)
            ax.axvline(direction, c='black')
            # for c in range(len(medians)):
            #     ax.axvline(medians[c], c=colors[c], linestyle='dotted')
            #     for tup in zip(*widths[c][1:]):
            #         print(tup)
            #         ax.hlines(tup[0], tup[1] * angular_res - 90, tup[2] * angular_res - 90, colors=colors[c], linestyle='--')
            #     axes[0].hlines(*widths[c][1:], color='red')
            # axes[0].plot(np.arange(16) * 180 / 15 - 90, 20 * np.log10(window), c='tab:green', marker='+', label='Hamming Window')
            ax.set_ylim(-40)
            ax.set_title('')
            ax.set_ylabel('Relative Power [dB]')
            ax.legend(loc='upper right')
            # plt.savefig(f'{plot_dir}/el_directivities.pdf', bbox_inches='tight')
            plt.show()
            plt.close()

    fig, ax = plt.subplots(1, 1, subplot_kw={'aspect': 'equal'})
    ax.plot(directions, directions, c='k')
    for d in [0, 2]:
        azimuths = np.array([diffs[d] for diffs in pointing_diffs])
        ax.plot(directions, np.array(azimuths), c=colors[d])
    ax.grid(True)
    ax.legend(['$\phi_d$', 'Main', 'Cross'])
    ax.set_xlim([-45, 0])
    ax.set_ylim([-45, 0])
    ax.set_xlabel('Nominal Receiver Beam Direction, $\phi_d$ [degrees]')
    ax.set_ylabel('Median Combined Beam Direction [degrees]')

    plt.savefig(f'{plot_dir}/adjusted_beam_directions.pdf', bbox_inches='tight')
    # plt.show()
    plt.close()

    nominal_directions = np.arange(24) * 3.24 - 37.26
    pointing_diffs = np.array(pointing_diffs)
    # print(pointing_diffs_2[:, -1])
    new_directions = np.round(np.interp(nominal_directions, pointing_diffs[:, -1], directions), decimals=1)
    print(f'XCFs: {new_directions}')
    new_directions = np.round(np.interp(nominal_directions, pointing_diffs[:, 0], directions), decimals=1)
    print(f'ACFs: {new_directions}')
