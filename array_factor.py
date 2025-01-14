import numpy as np
from scipy.constants import speed_of_light
import matplotlib.pyplot as plt

import plotting


def array_factor(weights, antenna_positions, freq, els, phis, el_factor=None):
    """Computes the array factor for the given array configuration.

    Parameters
    ----------
    weights: np.ndarray
        Array of weighting factors for each antenna. Each row is for one configuration
    antenna_positions: np.array
        Array of spatial positions of antennas. Array assumed to be linear. 1D
    freq: float
        Frequency of radio wave. Hz
    els: np.array
        Array of elevation points in degrees for 3D spatial grid. 1D
    phis: np.array
        Array of azimuthal points in degrees for 3D spatial grid. 1D

    Returns
    -------
    af: np.ndarray
        Array factor of radar array, at all points in [thetas, phis].
    """
    k = 2 * np.pi * freq / speed_of_light

    spatial_grid = k * np.outer(np.cos(els*np.pi/180), np.sin(phis*np.pi/180))
    exponents = np.einsum('i,jk->ijk', antenna_positions, spatial_grid)
    v = np.exp(1j * exponents)

    af = np.einsum('ij,jkl->ikl', weights, v) / len(antenna_positions)

    if el_factor is not None:
        af *= el_factor

    return 20 * np.log10(np.abs(af))


def default_spatial_arrays(resolution):
    """Creates elevation and azimuth arrays in radians with given angular resolution.

    Parameters
    ----------
    resolution: float
        Angular resolution in degrees.

    Returns
    -------
    els: np.array
        Array of elevation angles in degrees.
    azs: np.array
        Array of azimuth angles in degrees.
    """
    els = np.linspace(0, 90, round(90/resolution))
    azs = np.linspace(-90, 90, round(180/resolution))

    return els, azs


def default_antenna_positions(num_antennas, antenna_spacing: float = 15.24):
    """Makes the spatial positions of all antennas in the array.

    Assumes that all antennas are equally spaced in a linear array.

    Parameters
    ----------
    num_antennas: int
        Number of antennas in the array.
    antenna_spacing: float
        Uniform spacing between adjacent antennas. Meters.

    Returns
    -------
    antenna_positions: np.array
        Array of antenna positions along linear axis, in meters.
    """
    unit_spacing = np.linspace(-(num_antennas-1)/2, (num_antennas-1)/2, num_antennas)
    return antenna_spacing * unit_spacing


def uniform_weights(num_antennas):
    """Creates an array of uniform weights for each antenna.

    Parameters
    ----------
    num_antennas: int
        Number of antennas in array.

    Returns
    -------
    antenna_weights: np.array
        Array of weights for each antenna in the array.
    """
    return np.array(([1.0] * num_antennas))


def linear_phase(antenna_positions, freq, direction):
    """Creates a linear phase progression across the array.

    Parameters
    ----------
    antenna_positions: np.array
        Position of the antennas in the array. Assuming linear array.
    freq: float
        Frequency in Hz.
    direction: float
        Direction to "steer" the array to, in degrees right of boresight.

    Returns
    -------
    weights: np.array
        Array of weighting factors for each antenna.
    """
    k = 2 * np.pi * freq / speed_of_light
    direction_rad = direction * np.pi / 180.0

    angles = k * np.sin(-1 * direction_rad) * antenna_positions
    angles = np.fmod(angles, 2 * np.pi)

    return np.exp(1j * angles)


def ideal_pattern(phis):
    """Returns the ideal radiation pattern with the given angular points.

    Parameters
    ----------
    phis: np.array
        Azimuthal points in radians

    Returns
    -------
    pattern: np.array
        Ideal azimuthal radiation pattern.
    """
    pattern = np.zeros(phis.shape)
    passband = np.where(
        ((90 - 24.3) * np.pi / 180 < phis) & (phis < (90 + 24.3) * np.pi / 180))  # Mask for SuperDARN FOV

    pattern[passband] = 1.0 / 4

    return pattern


def kinsey_weights(taper, freq, passband, antenna_spacing):
    """Finds the proper beam-spoiling weights based on Kinsey, 1997 and Leifer, 2016.

    Parameters
    ----------
    taper: np.array
        Amplitude weights for each antenna in the array.
    freq: float
        Frequency in Hz.
    passband: tuple
        FOV of the array as (left bound, right bound) in degrees right of boresight.
    antenna_spacing: float
        Distance between antennas, in meters.

    Returns
    -------
    weights: np.array
        Complex weighting factors for each antenna.
    """
    normalization = np.dot(taper, taper)
    normalized_taper = taper / np.sqrt(normalization)
    taper_squared = np.multiply(normalized_taper, normalized_taper)
    taper_squared_cumsum = np.cumsum(taper_squared)

    wavelength = speed_of_light / freq

    lower_bound = passband[0]
    upper_bound = passband[1]
    k0 = 2 * np.pi / wavelength
    kn = np.zeros(len(taper) - 1)
    phases = np.zeros(len(taper))

    for i, mag in enumerate(taper_squared_cumsum[:-1]):
        kn[i] = np.sin((lower_bound + (upper_bound - lower_bound) * mag) * np.pi / 180) * k0
        phases[i+1] = phases[i] + kn[i] * antenna_spacing

    weights = np.exp(1j * phases)

    return weights * taper


def cached_genetic_weights(num_antennas, freq):
    cached_values_16_antennas = {
        10.4e6: [0., 33.21168501, 63.39856497, 133.51815213, 232.59694556, 287.65482653, 299.43588532, 313.30394893],
        10.5e6: [0., 33.22157987, 63.44769218, 134.09072554, 232.41818196, 288.18043116, 299.96678003, 312.81034918],
        10.6e6: [0., 33.49341546, 63.918406, 135.76673356, 232.41342064, 288.68373728, 299.8089564, 312.19755493],
        10.7e6: [0., 33.42706054, 63.94880958, 136.78441366, 232.43324622, 288.91978353, 299.57226291, 311.74840496],
        10.8e6: [0., 33.13909903, 63.56879316, 137.23017826, 232.17488475, 289.01436937, 299.53525025, 311.23785241],
        10.9e6: [0., 33.15305158, 63.55105706, 137.93590292, 232.13550152, 289.46328775, 299.78227805, 310.57614029],
        12.2e6: [0., 70.91038811, 122.60927618, 214.92179098, 276.38784179, 325.25390655, 351.3873793, 316.5693829],
        12.3e6: [0., 71.78224973, 124.29124213, 215.26781585, 277.84490172, 326.57004062, 353.22972278, 318.83181539],
        12.5e6: [0., 75.1870308, 128.12468688, 216.50545923, 281.26273571, 334.23044519, 357.70997722, 326.41420518],
        13.0e6: [0., 65.30441048, 122.04513377, 208.77532736, 282.14858123, 329.88094473, 368.67442895, 324.92709286],
        13.1e6: [0., 75.41723909, 133.59413156, 216.03815626, 287.94258174, 343.50035796, 369.91299149, 337.96682569],
        13.2e6: [0., 67.98474247, 126.21855408, 209.5839628, 285.48610109, 333.17276884, 370.37654775, 329.43903017],
        14.5e6: [0., 3.14970459, 92.61128586, 175.97789944, 220.42383933, 295.70931415, 290.2341177, 275.58267647]
    }
    cached_values_8_antennas = {
        10.4e6: [0., 25.65596691, 78.37293679, 139.64736262, 139.64736262, 78.37293679, 25.65596691, 0.],
        10.5e6: [0., 25.08958919, 77.59100768, 140.85808655, 140.85808655, 77.59100768, 25.08958919, 0.],
        10.6e6: [0., 24.57335302, 76.75481191, 141.98499171, 141.98499171, 76.75481191, 24.57335302, 0.],
        10.7e6: [0., 23.8098711, 75.90392693, 143.01444351, 143.01444351, 75.90392693, 23.8098711, 0.],
        10.8e6: [0., 22.11931133, 73.23562257, 143.47732068, 143.47732068, 73.23562257, 22.11931133, 0.],
        10.9e6: [0., 22.85211015, 72.76130323, 144.37536937, 144.37536937, 72.76130323, 22.85211015, 0.],
        12.2e6: [0., 24.12132192, 67.43277427, 160.59421469, 160.59421469, 67.43277427, 24.12132192, 0.],
        12.3e6: [0., 25.79888664, 68.32548572, 162.24856417, 162.24856417, 68.32548572, 25.79888664, 0.],
        12.5e6: [0., 29.73310292, 70.83940609, 166.04550735, 166.04550735, 70.83940609, 29.73310292, 0.],
        13.0e6: [0., 41.4313578, 82.16477044, 175.25809179, 175.25809179, 82.16477044, 41.4313578, 0.],
        13.1e6: [0., 43.20693263, 84.14234248, 175.38631445, 175.38631445, 84.14234248, 43.20693263, 0.],
        13.2e6: [0., 43.42908842, 84.21675093, 174.68458927, 174.68458927, 84.21675093, 43.42908842, 0.]
    }
    if num_antennas == 16:
        if freq in cached_values_16_antennas.keys():
            angles = cached_values_16_antennas[freq]
            phases = np.concatenate((angles, np.flip(angles)))
            weights = np.exp(-1j * np.deg2rad(phases))
        else:
            raise KeyError('Frequency not supported')
    elif num_antennas == 8:
        if freq in cached_values_8_antennas.keys():
            angles = cached_values_8_antennas[freq]
            weights = np.zeros(16, dtype=np.complex64)
            weights[4:12] = np.exp(-1j * np.deg2rad(angles))
        else:
            raise KeyError('Frequency not supported')
    else:
        raise ValueError('Number of antennas not supported')
    return weights


def plot_full_array_factor(gains):
    """Plot a color plot of the full array factor.

    Parameters
    ----------
    gains: np.ndarray
        Array of gain vs azimuth and elevation.
    """

    fig, ax1 = plt.subplots(1, 1, figsize=(12, 8))
    c = ax1.imshow(gains, origin='lower', aspect='auto')
    c.set_cmap('viridis')
    # c.set_clim([-25, 0])
    cbar = fig.colorbar(c, ax=ax1)
    cbar.set_label('Power (dB)')
    ax1.set_title('Full Array Factor')
    ax1.legend()
    plt.show()
    plt.close()


def plot_horizontal_gain(fig, ax, gains, angles, labels, colors, bounds=True):
    """Plot the gain pattern in the horizontal plane.

    Parameters
    ----------
    gains: list
        List of arrays with gain values. Array dimensions should match 'angles'
    angles: np.array
        Array of angular directions corresponding to gain patterns.
    labels: list(str)
        List of labels for each entry in 'gains'
    title: str
        Title of the plot.
    """
    #fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    #fig.suptitle(title)

    for i in range(len(gains)):
        ax.plot(angles, gains[i], label=labels[i], c=colors[i])

    if bounds:
        ax.axvline(x=-38.88, color='k', linestyle='--')
        ax.axvline(x=38.88, color='k', linestyle='--')
    ax.yaxis.grid()
    ax.set_xlabel('Azimuth [degrees]')
    ax.set_ylim([-40, 0])
    ax.set_ylabel('Normalized Array Factor [dB]')
    #ax.legend()
    #plt.savefig('/home/remington/kinsey_gain.pdf', bbox_inches='tight')
    #plt.close()


def plot_phases(fig, ax, weights, labels, legend=True):
    """Plots sets of antenna weights with associated labels.

    Parameters
    ----------
    ax: matplotlib.Axes
        Axis to plot on.
    weights: list of np.array
        List of arrays containing complex weights for each antenna.
    labels: list
        List of labels, one per set (row) in weights.
    title: str
        Title of the plot
    """
    if len(weights) != len(labels):
        raise ValueError("Could not match labels to data sets.")

    markers = ['*', '+', '+']
    for i in range(len(weights)):
        mask = np.abs(weights[i]) > 1e-3    # only plot active antennas
        print(np.abs(weights[i]))
        # ax1.plot(np.arange(weights[i].size)[mask], np.abs(weights[i][mask]), label=labels[i])  # plot amplitude
        ax.plot(np.arange(weights[i].size)[mask], np.unwrap(np.angle(weights[i]))[mask], label=labels[i], marker='+')

    major = plotting.Multiple(denominator=2)
    minor = plotting.Multiple(denominator=6)
    ax.yaxis.set_major_locator(major.locator())
    ax.yaxis.set_minor_locator(minor.locator())
    ax.yaxis.set_major_formatter(major.formatter())
    ax.yaxis.grid()

    # ax.set_xticks(range(1, max([len(w)+1 for w in weights]), 2), range(1, max([len(w)+1 for w in weights]), 2))     # every 2nd
    #ax1.set_ylabel("Magnitude")
    #ax1.legend()

    ax.set_xlabel("Antenna Index")
    ax.set_ylabel("Phase (rad)")
    if legend:
        ax.legend()
    # plt.savefig('/home/remington/Grad_Studies/bistatic_paper/antenna_phases.png', bbox_inches='tight')
    # plt.close()


def power_relative_to_full(weights):
    """Returns the power output of weights relative to a full amplitude array.

    Parameters
    ----------
    weights: np.array
        Array of complex weighting factors for each antenna

    Returns
    -------
    Power in dB relative to if each weight had magnitude 1.
    """
    return 10 * np.log10(np.sum(np.abs(weights), axis=1) / weights.shape[1])


def create_labels(labels, powers):
    """Add relative powers in dB to the label list.

    Parameters
    ----------
    labels: list
        List of labels for each quantity
    powers: list
        List of relative powers in dB for each quantity.

    Returns
    -------
    full_labels: list
        List of full labels
    """
    full_labels = [l + ' ({:.2f}dB)'.format(powers[i]) for i, l in enumerate(labels)]
    return full_labels
