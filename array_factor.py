import copy
import numpy as np
import scipy.linalg
from scipy.constants import speed_of_light
import scipy.optimize as optimize
import matplotlib.pyplot as plt

import plotting

#matplotlib.use('TkAgg')


def array_factor(weights, antenna_positions, freq, els, phis):
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


def parabolic_phase(num_antennas):
    """Creates a parabolic phase progression across the array.

    Parameters
    ----------
    num_antennas: int
        Number of antennas in the array.

    Returns
    -------
    phases: np.array
        Array of phase values for each antenna.
    """
    antennas = np.linspace(-(num_antennas-1)/2, (num_antennas-1)/2, num_antennas)
    angles = 8 * np.pi / ((num_antennas - 1) * (num_antennas - 1)) * np.multiply(antennas, antennas) - np.pi
    return np.exp(1j * angles)


def cosine_weighting(num_antennas):
    """Creates a cosine-window weighting across the array.

    Parameters
    ----------
    num_antennas: int
        Number of antennas in the array.

    Returns
    -------
    weights: np.array
        An amplitude weighting for each antenna in the array.
    """
    antennas = np.linspace(-(num_antennas-1)/2, (num_antennas-1)/2, num_antennas)
    return np.cos(antennas * np.pi/num_antennas)


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


def deviation_score(array_factor, phi_angles):
    """Computes the score of the array factor compared to the desired pattern.

    The score is a number which represents the closeness of the array factor to the
    desired widebeam pattern, with a score of 0 being perfect alignment. A lower
    score is better.

    Parameters
    ----------
    array_factor: np.array
        A vector of the array factor vs. azimuthal angle, from (0, pi).
    phi_angles: np.array
        A vector of the azimuthal angles corresponding to array_factor, in radians.

    Returns
    -------
    score: float
        Score of the array factor.
    """
    angular_res = phi_angles[1] - phi_angles[0]     # radians
    perfect_factor = ideal_pattern(phi_angles)

    passband = np.nonzero(perfect_factor)
    stopband = np.nonzero(perfect_factor == 0)      # all other indices

    ripple = np.max(array_factor[passband]) - np.min(array_factor[passband])

    diff = perfect_factor - array_factor
    square_integral = np.sqrt(np.sum(np.multiply(diff, diff)) * angular_res)

    sidelobe_level = np.min(array_factor[passband]) - np.max(array_factor[stopband])

    score = 1/(1 + np.exp(sidelobe_level)) + 2 * ripple #+ square_integral / 5
    return score


def random_deviation(weights, scale):
    """Randomly nudges a parameter of the weighting factor array.

    Parameters
    ----------
    weights: np.array
        Array of complex weighting factors for the array.
    scale: float
        Scaling factor for the random deviations.

    Returns
    -------
    weights: np.array
        Array of weighting factors with a nudged weight.
    """
    num_antennas = len(weights)
    half_antennas = int(np.ceil(num_antennas / 2))
    real_nudge = np.random.rand(half_antennas) * scale
    imag_nudge = np.random.rand(half_antennas) * scale * 1j
    # antenna = np.random.randint(0, np.ceil(num_antennas / 2))

    for antenna in range(half_antennas):
        mag = np.abs(weights[antenna] + real_nudge[antenna] + imag_nudge[antenna])
        if mag > 1.0:
            pass
        else:
            weights[antenna] += real_nudge[antenna] + imag_nudge[antenna]
            weights[num_antennas - antenna - 1] = weights[antenna]      # Keep the weights symmetric

    return weights


def find_weights(antenna_positions, angular_res, freq, max_iterations):
    """Finds the optimum set of complex weighting factors for the array.

    Parameters
    ----------
    antenna_positions: np.array
        Positions of the antennas in the array, in meters.
    angular_res: float
        Angular resolution of the coordinate system, in degrees.
    freq: float
        Frequency in Hz.
    max_iterations: int
        Maximum number of iterations.

    Returns
    -------
    weights: np.array
        Array of complex weighting factors for each antenna.
    """
    num_antennas = len(antenna_positions)
    best_weights = np.multiply(parabolic_phase(num_antennas), cosine_weighting(num_antennas))
    phis = np.linspace(0, 180, round(180/angular_res)) * np.pi / 180.
    num_iterations = 0
    score = 10
    prev_score = 10

    # Plot the weighting factors
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex='all')
    ax1.plot(range(num_antennas), np.abs(best_weights), label='Cosine')
    ax2.plot(range(num_antennas), np.angle(best_weights) * 180 / np.pi, label='Parabolic')

    while score > 0.2:
        if num_iterations == 0:
            weights = best_weights
        else:
            weights = random_deviation(copy.deepcopy(best_weights), 0.05)

        af = array_factor(weights, antenna_positions, freq, [90.0], phis)
        # af = 20 * np.log10(np.abs(af))
        af = np.abs(af)

        score = deviation_score(af[0, :], phis)

        if score < prev_score:
            best_weights = weights
            prev_score = score
            ax1.plot(range(num_antennas), np.abs(best_weights), label=num_iterations)
            ax2.plot(range(num_antennas), np.angle(best_weights) * 180 / np.pi, label=num_iterations)

        num_iterations += 1
        if num_iterations >= max_iterations:
            print("Iteration {}".format(num_iterations))
            break
    print("Score: {}".format(prev_score))

    ax1.set_ylabel("Magnitude")
    ax1.legend()
    ax2.set_xlabel("Antenna Index")
    ax2.set_ylabel("Phase (degrees)")
    ax2.legend()
    plt.show()

    return best_weights


def least_squares_weights(num_antennas, freq, passband, antenna_spacing, resolution):
    """Finds the least-squares solution for the antenna weights.

    The least-squares solution x of the equation Ax = b is computed, where b is the desired azimuthal
    gain of the system (1 in passband, 0 outside), x is the antenna weights, and A is the matrix of wavevectors for
    each antenna for all azimuthal directions.

    Parameters
    ----------
    num_antennas: int
        Number of equally spaced transmitting antennas.
    freq: float
        Frequency in Hz.
    passband: tuple
        FOV of the array as (left bound, right bound) in degrees CW of boresight.
    antenna_spacing: float
        Distance between antennas, in meters.
    resolution: float
        Azimuthal angular resolution, in degrees.

    Returns
    -------
    weights: np.array
        Complex weight for each antenna in the array.
    """
    azimuths = np.arange(-90, 90, resolution)

    b = np.ones(azimuths.shape)
    b[np.argwhere(passband[0] > azimuths)] = 0
    b[np.argwhere(azimuths > passband[1])] = 0

    k0 = 2 * np.pi * freq / speed_of_light

    antenna_indices = np.arange(num_antennas) - ((num_antennas - 1) / 2)
    wavevectors = np.sin(azimuths * np.pi / 180.0) * k0

    # An array of wavevectors for each antenna and azimuth. Dimensions [num_azimuths, num_antennas]
    arr = np.outer(wavevectors, antenna_indices)

    A = np.exp(-1j * arr * antenna_spacing)

    weights, _, _, _ = scipy.linalg.lstsq(A, b)
    max_mag = np.max(np.abs(weights))
    weights = weights / max_mag

    print(np.max(np.abs(weights)))
    print(np.min(np.abs(weights)))
    for w in weights:
        print('{:.2f}'.format(np.abs(w)))
    print('Normalized power: {}'.format(np.sum(np.abs(weights)) / num_antennas))

    return weights


def uniform_optimizer(num_antennas, freq, passband, antenna_spacing, resolution):
    """
    Computes the ideal weighting factors for each antenna subject to the constraint that the magnitude of each
    weight is equal to one. This function uses the method outlined in 'Magnitude Least–Squares Fitting via Semidefinite
    Programming with Applications to Beamforming and Multidimensional Filter Design', by Peter Kassakian (2005)
    https://www.cnmat.berkeley.edu/sites/default/files/attachments/2005_Magnitude_Least--Squares_Fitting_via_Semidefinite_Programming.pdf

    Parameters
    ----------
    num_antennas: int
        Number of linearly spaced antennas in the array.
    freq: float
        Frequency in Hz.
    passband: tuple
        Tuple of (left bound, right bound) of the desired FOV.
    antenna_spacing: float
        Uniform inter-element spacing of the antenna array.
    resolution: float
        Azimuthal angular resolution, in degrees.

    Returns
    -------
    weights: np.array
        A set of complex weighting factors, one per antenna.
    """
    azimuths = np.arange(-90, 90, resolution)
    m = azimuths.size

    b = np.ones(azimuths.shape)
    b[np.argwhere(passband[0] > azimuths)] = 0
    b[np.argwhere(azimuths > passband[1])] = 0
    B = np.identity(m) * b

    k0 = 2 * np.pi * freq / speed_of_light
    antenna_indices = np.arange(num_antennas) - ((num_antennas - 1) / 2)
    wavevectors = np.sin(azimuths * np.pi / 180.0) * k0

    # An array of wavevectors for each antenna and azimuth. Dimensions [num_azimuths, num_antennas]
    arr = np.outer(wavevectors, antenna_indices)

    A = np.exp(-1j * arr * antenna_spacing)

    # Some convenience matrices
    temp1 = np.matmul(A.conj().T, A)
    temp2 = np.matmul(A, np.linalg.inv(temp1))
    temp3 = np.matmul(temp2, A.conj().T)
    U = temp3 - np.identity(A.shape[0])
    W = np.matmul(np.matmul(U, B).conj().T, np.matmul(U, B))
    W_tilde = np.zeros((2*m, 2*m))
    W_tilde[:m, :m] = np.real(W)
    W_tilde[:m, m:] = -1 * np.imag(W)
    W_tilde[m:, :m] = np.imag(W)
    W_tilde[m:, m:] = np.real(W)

    C = cvxpy.Variable((2*m, 2*m))
    constraints = [C >> 0]
    constraints += [C[i,i] + C[i+m,i+m] == 1 for i in range(m)]
    prob = cvxpy.Problem(cvxpy.Minimize(cvxpy.trace(C @ W_tilde)), constraints)
    prob.solve()

    C_opt = C.value
    num_iterations = 100
    min_val = 0
    for i in range(num_iterations):
        s_bar = np.random.multivariate_normal(np.zeros(2*m), C_opt)
        s_c = s_bar[:m] + 1j * s_bar[m:]
        np.divide(s_c, np.abs(s_c))
        score = np.matmul(np.matmul(s_c.conj().T, W), s_c)
        if i == 0:
            min_val = score
            s_opt = s_c
        elif score < min_val:
            min_val = score
            s_opt = s_c

    weights = np.matmul(np.matmul(np.matmul(temp1, A.conj().T), B), s_opt)

    return weights / np.max(np.abs(weights))


def phase_only_synthesis(num_antennas, freq, passband, transition_width, psl, ripple, antenna_spacing, resolution):
    """
    Computes the ideal weighting factors for each antenna, fixing the magnitude of each weight to 1.
    This method uses some relaxation methods to find an ideal solution which matches a given passband ripple and
    maximum sidelobe level. Taken from 'Phase-Only Pattern Synthesis for Linear Antenna Arrays' by Liang et al. (2017).
    https://ieeexplore-ieee-org.cyber.usask.ca/document/8103884/references#references


    Parameters
    ----------
    num_antennas: int
        Number of linearly spaced antennas in the array.
    freq: float
        Frequency in Hz.
    passband: tuple
        Tuple of (left bound, right bound) of the desired FOV.
    transition_width: float
        Width of transition region in degrees.
    psl: float
        Maximum sidelobe level in dB down from passband.
    ripple: float
        Maximum deviation from 0 dB within the passband, in dB.
    antenna_spacing: float
        Uniform inter-element spacing of the antenna array.
    resolution: float
        Azimuthal angular resolution, in degrees.

    Returns
    -------
    weights: np.array
        A set of complex weighting factors, one per antenna.
    """
    upper_ripple = np.power(10, ripple / 10)
    lower_ripple = np.power(10, -ripple / 10)

    pb_azimuths = np.arange(passband[0], passband[1], resolution)
    sb_low_azimuths = np.arange(-90, passband[0] - transition_width, resolution)
    sb_high_azimuths = np.arange(passband[1] + transition_width, 90, resolution)
    N = pb_azimuths.size + sb_low_azimuths.size + sb_high_azimuths.size

    # Create the upper and lower bounds for the solution at any given azimuthal direction.
    upper_bound = np.ones(N) * upper_ripple
    upper_bound[pb_azimuths.size:] = np.power(10, -psl/10)

    lower_bound = np.ones(N) * lower_ripple
    lower_bound[pb_azimuths.size:] = 0

    azimuths = np.concatenate((pb_azimuths, sb_low_azimuths, sb_high_azimuths))
    k0 = 2 * np.pi * freq / speed_of_light
    antenna_indices = np.arange(num_antennas) - ((num_antennas - 1) / 2)
    wavevectors = np.sin(azimuths * np.pi / 180.0) * k0

    # An array of wavevectors for each antenna and azimuth. Dimensions [num_antennas, num_azimuths]
    a = np.outer(antenna_indices, wavevectors)
    a = np.exp(-1j * a * antenna_spacing)

    # Initialize the parameters that get iteratively updated.
    xi = np.random.uniform()
    phi = np.random.uniform(size=(num_antennas, 1)) * 2 * np.pi
    lam = np.zeros((1, N))
    A = np.identity(num_antennas)
    rho = 1

    steering_vec = np.matmul(A, a)
    a_H = a.conj().T

    x = xi * np.matmul(np.exp(-1j * phi.T), steering_vec)

    num_iterations = 300

    # Iterate a set number of times to reach a solution.
    for t in range(num_iterations):
        # Calculate new values for x
        x_tilde = x - lam / rho
        mag_sq = np.abs(x_tilde) * np.abs(x_tilde)
        for i in range(x.size):
            if mag_sq[0, i] >= upper_bound[i]:
                x[0, i] = np.sqrt(upper_bound[i]) * np.exp(1j * np.angle(x_tilde[0, i]))
            elif mag_sq[0, i] <= lower_bound[i]:
                x[0, i] = np.sqrt(lower_bound[i]) * np.exp(1j * np.angle(x_tilde[0, i]))
            else:
                x[0, i] = x_tilde[0, i]

        def calculate_EF(phi):
            """Calculate values E and F from the paper cited above."""
            temp1 = np.matmul(np.exp(-1j * phi.T), steering_vec)
            temp2 = np.matmul(A, np.exp(1j * phi))
            temp3 = np.matmul(a_H, temp2)
            E = rho / 2 * np.sum(np.matmul(temp1, temp3))

            temp4 = np.matmul((x + lam/rho).conj(), temp1)
            temp5 = np.matmul((x + lam/rho), temp3)
            F = 1 - (rho / 2) * (np.sum(temp4) + np.sum(temp5))
            return E, F

        def cost_function(phi):
            """Returns a scalar cost value based on the paper cited above."""
            E, F = calculate_EF(phi)
            return -(F * F) / (4 * E)

        result = optimize.minimize(cost_function, phi, method='BFGS')
        if result.success:
            phi = result.x
        else:
            raise RuntimeError("Could not converge to a value.")

        E, F = calculate_EF(phi)
        xi = -F / (2 * E)
        lam_real = np.real(lam) + rho * np.real(x - xi * np.matmul(np.exp(-1j * phi), steering_vec))
        lam_imag = np.imag(lam) + rho * np.imag(x - xi * np.matmul(np.exp(-1j * phi), steering_vec))
        lam = lam_real + 1j * lam_imag

    weights = np.matmul(A, np.exp(1j * phi))
    return weights


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


def quadratic_beamspoiling(num_antennas, antenna_spacing, freq):
    """Returns the quadratic beamspoiling applied to the tapers.

    Parameters
    ----------
    num_antennas: int
        Number of antennas in the array.
    antenna_spacing: float
        Uniform antenna spacing, in meters.
    freq: float
        Frequency in Hz.

    Returns
    -------
    phases: np.array
        Complex phases for each antenna.
    """
    phis = np.zeros(num_antennas)
    antenna_indices = np.linspace(-(num_antennas-1)/2, (num_antennas-1)/2, num_antennas)
    wavelength = speed_of_light / freq
    delta_k = 2 * np.pi * np.sin(np.pi * 2 * 24.3 / 180) / wavelength

    for idx, i in enumerate(antenna_indices):
        phis[idx] = i*i * antenna_spacing * delta_k / (2 * (num_antennas - 1))

    return np.exp(1j * phis)


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


def plot_horizontal_gain(fig, ax, gains, angles, labels):
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
        ax.plot(angles, gains[i], label=labels[i])

    ax.axvline(x=-24.3-3.24/2, color='k', linestyle='--')
    ax.axvline(x=24.3+3.24/2, color='k', linestyle='--')
    ax.yaxis.grid()
    ax.set_xlabel('Azimuth (degrees)')
    # ax.set_ylim([-25, 0])
    ax.set_ylabel('Normalized Array Factor (dB)')
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


def plot_surface(thetas, phis, array_factor):
    """Plots a 3D surface of the array factor in all directions.

    Parameters
    ----------
    thetas: np.array
        Array of zenith angles in degrees.
    phis: np.array
        Array of azimuthal angles in degrees.
    array_factor: np.ndarray
        Array of array factor magnitudes for (theta, phi) points.
    """
    theta, phi = np.meshgrid(phis * np.pi/180, thetas * np.pi/180)
    x = np.ravel(array_factor * np.sin(theta) * np.cos(phi))
    y = np.ravel(array_factor * np.sin(theta) * np.sin(phi))
    z = np.ravel(array_factor * np.cos(theta))

    from matplotlib.tri import Triangulation
    tri = Triangulation(np.ravel(theta), np.ravel(phi))
    plt.figure()
    ax = plt.axes(projection='3d')
    ax.plot_trisurf(x, y, z, triangles=tri.triangles, cmap='viridis', linewidths=0.2)
    plt.show()


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


def main():
    num_antennas = 16   # main array
    freq = 10.8e6       # Hz
    angular_res = 0.5   # degrees
    left_bound = -24.3  # FOV boundary, in degrees right of boresight
    right_bound = 24.3  # FOV boundary, in degrees right of boresight
    direction = 0       # degrees right of boresight
    elevation = 0       # degrees up from horizon
    wavelength = speed_of_light / freq
    antenna_spacing = 15.24     # meters
    max_iterations = 1000
    weights = []
    labels = []

    # Useful constants
    k = 2 * np.pi * freq / speed_of_light
    direction_rad = direction * np.pi / 180.0

    # Position of each antenna along y-axis
    antenna_positions = default_antenna_positions(num_antennas, antenna_spacing)

    # Arrays of spherical coordinate points
    # els, azimuths = default_spatial_arrays(angular_res)
    els = np.array([elevation])
    azimuths = np.arange(-90, 90, angular_res)

    # Search for the optimum weighting factors.
    # weights = find_weights(antenna_positions, angular_res, freq, max_iterations)

    passband = (left_bound - direction, right_bound - direction)

    def add_configuration(w, label):
        """Add a set of weights with associated label for simulation."""
        weights.append(w)
        labels.append(label)

    # gs = GeneticSolver(num_antennas, antenna_spacing, freq, 1.0, -6, passband, 5, 200, 200)
    # add_configuration(gs._weights, 'Genetic')
    # add_configuration(phase_only_synthesis(num_antennas, freq, passband, 5, 14, 1.5, antenna_spacing, angular_res), 'POAPS')
    # add_configuration(uniform_optimizer(num_antennas, freq, passband, antenna_spacing, 1.0), 'SDP Optimized')
    # add_configuration(least_squares_weights(num_antennas, freq, passband, antenna_spacing, angular_res), 'Least Squares')
    # add_configuration(kinsey_weights(windows.hamming(num_antennas), freq, passband, antenna_spacing), 'Kinsey-Hamming')
    # add_configuration(uniform_weights(num_antennas), 'Standard Beamforming')
    #add_configuration(kinsey_weights(uniform_weights(num_antennas), freq, passband, antenna_spacing), 'Kinsey')
    weights_8 = kinsey_weights([0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0], freq, passband, antenna_spacing)
    ref = np.exp(-1j * np.angle(weights_8[4]))
    add_configuration(weights_8 * ref, 'Kinsey')
    #add_configuration(kinsey_weights([0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0], freq, passband, antenna_spacing), 'Kinsey')
    #add_configuration(uniform_weights(num_antennas), 'Standard Beamforming')
    #normal = cached_genetic_weights(16, freq)
    #add_configuration(normal, 'Genetic Solver')

    # antenna_10_down = copy.deepcopy(normal)
    # antenna_10_down[10] = 0.0
    # antennas_4_and_10_down = copy.deepcopy(normal)
    # antennas_4_and_10_down[4] = 0.0
    # antennas_4_and_10_down[10] = 0.0
    # add_configuration(antenna_10_down, 'INV')
    # add_configuration(antennas_4_and_10_down, 'RKN')

    add_configuration(cached_genetic_weights(8, freq), 'Genetic')
    #add_configuration([0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0], '2 Antennas')
    #add_configuration(kinsey_weights(uniform_weights(num_antennas), freq, passband, antenna_spacing), 'Kinsey')
    add_configuration(cached_genetic_weights(16, freq), '16 Antennas')
    add_configuration(uniform_weights(num_antennas), 'Standard Beamforming')
    # add_configuration([0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0], 'Three Antennas')

    weights = np.array(weights)

    # Compute the array factor for each set of weights.
    af = array_factor(weights, antenna_positions, freq, els, azimuths)

    # Plot the array pattern
    # plot_full_array_factor(af[0, ...])

    gains = [af[i, 0, :] for i in range(af.shape[0])]
    powers = power_relative_to_full(weights)

    full_labels = labels #create_labels(labels, powers)

    # Plot the zero-elevation array factor
    #title = 'Array Factor at {:.1f} MHz'.format(freq * 1e-6)
    #plot_horizontal_gain(gains, azimuths, full_labels, title)

    # Plot the complex weights for each antenna
    title = 'Relative Phases at {:.1f} MHz'.format(freq * 1e-6)
    fig, axes = plt.subplots(2, 1, figsize=(6, 6))
    fig.tight_layout()
    plot_horizontal_gain(fig, axes[0], gains, azimuths, labels)
    plot_phases(fig, axes[1], weights, labels)
    axes[0].set_ylim(-40)
    axes[1].set_title('')
    # plt.savefig('/home/remington/8_antennas_all.pdf', bbox_inches='tight')
    plt.show()
    plt.close()
    # plot_surface(els, azimuths, af)


if __name__ == '__main__':
    main()
