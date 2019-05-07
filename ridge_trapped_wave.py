import sys
from warnings import filterwarnings, catch_warnings, warn
import numpy as np
import numpy.ma as ma
from scipy.linalg import eig, eigvals
from scipy.integrate import simps
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt


def midpoint(x, axis=0):
    return x[:-1] + np.diff(x, axis=axis)/2


def remove_duplicate_depths(x, depths, print_progress=True):
    dup_idx = np.isclose(depths[:-1], depths[1:])
    if any(dup_idx) and print_progress:
        warn_msg = (str(sum(dup_idx)) + ' adjacent duplicate depths in input'
                    ' have been removed\n')
        warn(warn_msg, RuntimeWarning)
    x = x[~dup_idx]
    depths = depths[~np.append(dup_idx, False)]
    return x, depths


def calc_non_rotating_speed(N, z, nmodes):
    """Mode-1 non-rotating internal wave speed

    Ported to Python from Sam Kelly's MODES function in his CELT model

    http://www.d.umn.edu/~smkelly/software.html
    """
    N, z = np.asanyarray(N), np.asanyarray(z)
    H = z.max()

    N_intp = interp1d(-z, N, fill_value='extrapolate')

    nz = 5*nmodes + 1  # Number of vertical grid points
    dz = H/nz
    z = -np.r_[0:H:nz*1j]

    D2 = (np.diag(-2*np.ones(nz-1)/dz**2) +
          np.diag(np.ones(nz-2)/dz**2, 1) +
          np.diag(np.ones(nz-2)/dz**2, -1))

    # Boundary conditions
    D2[0, 0] = -2/dz**2
    D2[0, 1] = 1/dz**2
    D2[-1, -1] = -2/dz**2
    D2[-1, -2] = 1/dz**2

    A = np.diag(-N_intp(midpoint(z))**2)

    k2 = eigvals(D2, A)
    c = np.max(np.real(1/np.sqrt(k2)))

    return c


def calc_vertical_modes(N, N0, z, h, nmodes):
    """Solve Sturm-Liouville problem for vertical modes and eigenenvalues

    Broadly follows page 2677 of Durran (2000) doi:10.1002/qj.49712656904
    with changes to apply to ocean not atmosphere (i.e., upper boundary
    condition)

    Inputs
    ------
    N: array-like
        Buoyancy frequency vector at points z
    z: array-like
        | Same length as N and goes to deepest depth exactly
        | z values are passed in as positive depths
    h: float
        Non-dimensionalised depth (0 < h < 1)
    nmodes: int
        Number of modes to return

    Returns
    -------
    gamma_n_sq_minus_lhat_sq: array
        nz - 2 eigenvalues for the vertical modes. What Durran (2000) terms
        alpha_n^2 - lambda^2
    phi_n: list
        | nz - 2 functions that give the eigenvectors
        | For example, phi_n[2](z) is the mode-2 eigenvector where 0 < z < 1
        | Note that eigenvector amplitudes are scaled to give desired
        | P_mn when eigvec^2 is integrated over 0 to h
    """

    N, z = np.asanyarray(N), np.asanyarray(z)

    nz = 5*nmodes + 1  # Number of vertical grid points
    dz = h*1/(nz + 1)

    N_tilde = interp1d(-z/z.max(), N/N0, fill_value='extrapolate')

    z = np.cumsum(dz*np.ones(nz))
    # Extend z beyond range so that quadratic extrapolation later works better
    zp5 = midpoint(np.r_[0, z, h, h + dz])
    zm5 = midpoint(np.r_[-dz, 0, z, h])  # z - 0.5
    z = np.r_[0, z, z[-1] + dz]

    # Flip problem upside down so that z is negative
    z, zp5, zm5 = -z, -zp5, -zm5

    LHS = np.diag(-1/N_tilde(zp5)**2 - 1/N_tilde(zm5)**2)
    LHS += np.diag(1/N_tilde(zp5[:-1])**2, k=1)
    LHS += np.diag(1/N_tilde(zm5[1:])**2, k=-1)
    LHS /= dz**2

    # Boundary condition from second-order, one-sided derivative
    LHS[0, :3] = 3, -4, 1
    LHS[-1, -3:] = 1, -4, 3

    # First and last rows are boundary condition
    # Account for this by making RHS zero for these two
    RHS = -np.diag(np.r_[0, np.ones(nz), 0])

    eigenvals, eigenvecs = eig(LHS, RHS)
    # Occasionally eigenvalues come out negative, but need to be positive
    eigenvals = np.abs(eigenvals)
    # Remove complex parts of eigenvectors, which is zero anyway
    eigenvecs = np.real(eigenvecs)

    good_vals = np.where(np.isfinite(eigenvals))[0]
    eigenvals, eigenvecs = np.real(eigenvals[good_vals]), eigenvecs[:, good_vals]

    idx = np.argsort(eigenvals)
    eigenvals, eigenvecs = eigenvals[idx], eigenvecs[:, idx]

    # Only first M modes should be returned
    eigenvals, eigenvecs = eigenvals[:nmodes], eigenvecs[:, :nmodes]

    # Scale eigenvecs such that P_mn gives same result as constant N case
    # For the linear case, this sets amplitude to 1 for all vectors
    eigenvecs /= np.sqrt(simps(eigenvecs**2, dx=dz, axis=0)*2/h)
    eigenvecs[:, 0] *= np.sqrt(2)

    # Make eigenvecs all positive at the seafloor. (Output from eig gives
    # functions an arbitrary sign). This step isn't strictly necessary, but
    # keeps output closer to analytical result
    eigenvecs *= np.sign(eigenvecs[-1, :])

    # Rename and adjust output to something more useful
    gamma_n_sq_minus_lhat_sq = eigenvals
    phi_n = [interp1d(z, eigenvec, fill_value='extrapolate', kind='quadratic')
             for eigenvec in eigenvecs.T]

    return gamma_n_sq_minus_lhat_sq, phi_n


def P_mn(hj, nmodes=20):
    P_mn = np.matrix(hj/2*np.eye(nmodes))
    P_mn[0, 0] = hj
    return P_mn


def Q_mn(h1, h2, nmodes=20):
    m, n = np.mgrid[:nmodes, :nmodes]

    with catch_warnings():
        filterwarnings('ignore', '.*divide by zero*.')
        filterwarnings('ignore', '.*invalid value encountered in true_divide*.')
        numerator = h1**2*h2*m*np.sin(m*np.pi*(h1-h2)/h2)
        denominator = h1**2*m**2*np.pi-h2**2*n**2*np.pi
        Q_mn = np.matrix(numerator/denominator)

        # Deal with singularity
        sing = np.isclose(m*h1, n*h2)
        Q_mn[sing] = ((-1)**n/2*(h1*np.cos(h2*n*np.pi/h1)))[sing]

        Q_mn[0, 0] = h1
    return Q_mn


def P_mn_variable_N(h, vecs):
    """Integrals from -h to 0 of all combinations of eigenvectors

    This comes out very close to P_mn from the analytical solution

    But imperfect orthogonality may be important

    Returns
    -------
    Matching matrix
    """
    nmodes = len(vecs)
    P_mn = np.zeros((nmodes, nmodes))
    z = np.r_[-h:0:101j]
    dz = np.diff(z).mean()
    for i, j in np.ndindex(nmodes, nmodes):
        vec_pair = vecs[i](z)*vecs[j](z)
        P_mn[i, j] = simps(vec_pair, dx=dz)

    return np.matrix(P_mn.T)


def Q_mn_variable_N(h1, h2, vecs1, vecs2):
    """Integrals from -h1 to 0 of all combinations of eigenvectors

    Returns
    -------
    Matching matrix
    """
    nmodes = len(vecs1)
    Q_mn = np.zeros((nmodes, nmodes))
    z = np.r_[-h1:0:101j]
    dz = np.diff(z).mean()
    for i, j in np.ndindex(nmodes, nmodes):
        vec_pair = vecs1[i](z)*vecs2[j](z)
        Q_mn[i, j] = simps(vec_pair, dx=dz)

    return np.matrix(Q_mn.T)


class RidgeTrappedWave(object):
    """Derive properties of an subinertial wave beside a ridge, step, or coast

    This class is called RidgeTrappedWave because that's what it was
    originally designed for. However, the same core code also works for steps
    and ridges.

    See Hughes et al. (2018) Tidal conversion and dissipation at steep
    topography in a channel poleward of the critical latitude. In prep for
    J. Phys. Oceanogr.

    Specifically, iteratively solve an eigenvalue problem that arises by
    matching pressure and velocity at each step discontinuity.

    Why iteratively? omega is the eigenvalue of the problem, but omega is
    also used in construction of the LHS and RHS matrices.

    It is assumed that omega is known, and wavenumber `l` is unknown,
    so in each iteration `l` is adjusted

    Inputs
    ------
    x: 1D array
        Array defining the locations of the discontinuites (metres)
    depths: 1D array
        | Array defining the depths (metres). This must be one element larger
        | than ``x``
        | For a coastal trapped wave (ie not a step or ridge) the last depth
        | should be identically zero
    omega: float
        | Non-dimensionalised tidal forcing frequency between 0 and 1
        | omega is dimensional frequency divided by coriolis frequency
    N: float or two-column vector
        | Buoyancy frequency (rad/s)
        | For constant N case, only a single value is needed
        | For variable N case, first column is values of N and second column
        | is associated depths. Linear interpolation is used between points
        | See examples for further clarification
    lat: float
        | Latitude in decimal degrees (needed for Coriolis frequency)
        | May not work for negative latitudes
    lambda_guess: float (optional)
        | First estimate of the wavelength (m)
        | This may speed up iteration, but convergence is usually rapid and
        | this parameter can often be safely ignored
    nmodes: int
        Number of vertical modes (default of 20).
    mode: int
        | Which mode to solve for. Defaults to lowest mode (0)
        | Python counts from 0, so ``mode = 0`` is the lowest baroclinic mode
        | Only one mode can be solved for at a time given the iterative nature
        | of the problem
    print_progress: bool
        Whether to provide updates regarding iteration

    Usage Notes
    -----------

    - The distance between steps should ideally be a small fraction of the
      internal Rossby radius
    - For omega very close to 1 (i.e., greater than 0.95), plots of U and V
      will be noisy.
    - The internal Rossby radius for the problem is calculated from the
      leftmost depth. Therefore, the left side of the ridge should be equal
      to or deeper than the seafloor on the right side of the ridge.
    - Code may or may not work when negative latitudes are used. Just use
      positive ones.
    - The main eigenvalue problem involves M × M matrices, where M is
      2 × number of modes × number of steps. Hence, when the number of steps
      gets large (> 50), calculating the solution may take several minutes.
    - The coastal problem (i.e., where depth goes to zero) has undergone only
      limited testing.
    - If the solution doesn't clearly converge, try a different value for `lambda_guess`

    Examples
    --------
    Constant stratification

    >>> x = 0, 3000
    >>> z = 250, 50, 250
    >>> omega = 0.9
    >>> N = 6E-3
    >>> lat = 80
    >>> r = RidgeTrappedWave(x, z, omega, N, lat, lambda_guess=40e3)
    >>> r.calc_wavelength_and_coefficients(niter_max=8)
    >>> # r.contour_mode_shapes() # Plot solution
    >>> print(r)
    Iterating
    λ (km)      ω
    40.0
    44.4        0.917
    46.12       0.9047
    46.13       0.9001
    Converged to specified accuracy
    Now calculating eigenmode
    Ridge Trapped Wave with wavelength of 46.1 km

    Vertically variable stratification

    >>> x = 0, 3000
    >>> z = 250, 50, 250
    >>> omega = 0.9
    >>> N = [[8E-3, 0], [6E-3, 125], [2E-3, 250]]
    >>> lat = 80
    >>> r = RidgeTrappedWave(x, z, omega, N, lat, lambda_guess=40e3)
    >>> r.calc_wavelength_and_coefficients(niter_max=8)
    >>> # r.contour_mode_shapes() # Plot solution
    >>> print(r)
    Calculating vertical modes and matching matrices
    Iterating
    λ (km)      ω
    40.0
    44.4        0.924
    49.2        0.912
    49.27       0.9002
    Converged to specified accuracy
    Now calculating eigenmode
    Ridge Trapped Wave with wavelength of 73.0 km

    Coastal trapped wave - compare with analytical internal Kelvin wave

    >>> x = 0, 10, 20  # Very closely spaced steps to approximate wall
    >>> z = 250, 200, 100, 0  # This is minimum no. depths for costal problem
    >>> omega = 0.9
    >>> N = 6E-3
    >>> lat = 80
    >>> r = RidgeTrappedWave(x, z, omega, N, lat)
    >>> r.calc_wavelength_and_coefficients()
    Iterating
    λ (km)      ω
    38.7
    43          0.63
    17          0.58
    20.9        0.963
    23.4        0.925
    22.8        0.889
    22.75       0.8990
    22.75       0.9000
    Analytical solution is 2*N*max(z)/(omega*f) = 23.2 km
    """

    def __init__(
            self, x, depth, omega, N, lat, lambda_guess=None,
            nmodes=20, mode=0, print_progress=True):

        # Check and adjust inputs before doing anything
        x = np.asanyarray(x)
        depth = np.asanyarray(depth)
        N = np.asanyarray(N)
        too_few_steps_msg = (
            'This code does not work for a single step or less.\n'
            'x and z vectors have minimum size of 2 and 3, respectively,\n'
            'for ridge or step or 3 and 4, respectively, for coastal problem')
        assert (x.size > 1 and depth[-1] != 0) or x.size > 2, too_few_steps_msg
        assert len(depth) == len(x) + 1, 'len(depths) must equal len(x) + 1'
        if all(depth < 0):
            warn('Depths should be positive. Changing sign', RuntimeWarning)
            depth *= -1
        assert 0 < omega < 1, '0 < omega < 1 (omega is non-dimensionalised by f)'
        assert np.all(np.diff(x) > 0), 'x must increase monotonically'
        assert np.array(N).size == 1 or np.array(N).shape[1] == 2, (
            'N must be either a single value or two-column array')
        x, depth = remove_duplicate_depths(x, depth, print_progress)
        if len(x)*nmodes > 500:
            M = str(2*len(x)*nmodes)
            warn('Matrices are large (' + M + ' x ' + M + ')\n'
                 'Computing eigenmode may take several minutes', RuntimeWarning)
            sys.stderr.flush()

        if np.isclose(depth[-1], 0):
            self.is_coastal_wave = True
            # Remove last x, depth pair and keep track of that last dx
            self.L_endx2 = np.diff(x[-2:])
            x = x[:-1]
            depth = depth[:-1]
        else:
            self.is_coastal_wave = False

        self.x = x
        self.depth = depth
        self.omega = omega
        self.omega_in = 1.0*omega
        self.print_progress = print_progress
        if np.array(N).size == 1:
            self.N0 = N
            self.N = N
            self.variable_N = False
        else:
            self.N = N[:, 0]
            self.zN = N[:, 1]
            c = calc_non_rotating_speed(self.N, self.zN, nmodes)
            self.N0 = np.pi*c/self.depth.max()
            self.variable_N = True
        self.lat = lat
        self.h = depth/depth.max()
        self.lat_to_f(lat)
        self.Lr = self.N0*self.depth.max()/(self.f*np.pi)
        if lambda_guess is None:
            lambda_guess = (
                2*self.N0*depth.max()/(omega*self.f))/(1-self.h.min())
            self.lambda_guess = lambda_guess/(mode+1)
        else:
            self.lambda_guess = lambda_guess
        self.l = 2*np.pi/self.lambda_guess
        self.nmodes = nmodes
        self.mode = mode

        self.jvec()
        self.set_R()
        self.update_horizontal_scale()
        if self.variable_N:
            if print_progress:
                print('Calculating vertical modes and matching matrices')
            self.calc_variable_N_modes()
            self.eval_matching_matrices_variable_N(
                self.h, self.phi_nj, self.nmodes)
        else:
            self.eval_matching_matrices_constant_N(self.h, self.nmodes)
        self.wavelength_str = ('UNDETERMINED. Run calc_wavelength_and_coefficients '
                               'method to determine')

    def __repr__(self):
        x_str = '[' + ', '.join(str(np.round(
            xi, decimals=1)) for xi in self.x) + ']'
        depth_str = '[' + ', '.join(str(np.round(
            zi, decimals=1)) for zi in self.depth) + ']'
        if self.variable_N:
            N_str = '['
            for Ni, zi in zip(self.N, self.zN):
                N_str += '[{0:2.3e}, {1:2.1f}], '.format(Ni, zi)
            N_str = N_str[:-2] + ']'
        else:
            N_str = self.N0
        fmt_str = (
            '{}(\n'
            '    x={},\n'
            '    depth={},\n'
            '    omega={!s},\n'
            '    N={},\n'
            '    lat={!s},\n'
            '    lambda_guess={!s},\n'
            '    nmodes={!s},\n'
            '    mode={!s})')
        return fmt_str.format(
            self.__class__.__name__,
            x_str, depth_str, self.omega_in, N_str,
            self.lat, self.lambda_guess, self.nmodes, self.mode)

    def __str__(self):
        return (wave_type + ' Trapped Wave with wavelength of ' +
            self.wavelength_str)


    def lat_to_f(self, lat):
        self.f = 2*7.29211E-5*np.sin(np.deg2rad(lat))

    def jvec(self):
        """Total number of discontinuties and helpful counter"""
        self.J = len(self.x)
        self.js = np.r_[:self.J+1]

    def set_R(self):
        """R is defined by Chapman (1982) doi:10.1016/0377-0265(82)90002-1"""
        # See equation 4 for unapproximated R (not value used in Appendix B)
        self.R = ((self.N0/self.f) - self.omega**2)/np.sqrt(1-self.omega**2)

    def update_horizontal_scale(self):
        """Non-dimensionalise by H and further scale by R

        Scale changes during iteration
        """
        self.xhat = self.x/(self.R*self.depth.max())
        self.lhat = self.l*self.R*self.depth.max()
        Lj_hat = np.diff(self.xhat)/2
        # Make Lj same size as other _j vectors
        self.Lj_hat = np.r_[np.nan, Lj_hat, np.nan]

    def calc_variable_N_modes(self):
        """For each flat-bottomed region, calculate all vertical modes
        and associated eigenvalues"""
        # gn2_m_lh2 is gamma_n^2 - lhat^2
        self.gn2_m_lh2 = [[] for _ in self.js]
        self.phi_nj = [[] for _ in self.js]
        for j in self.js:
            self.gn2_m_lh2[j], self.phi_nj[j] = calc_vertical_modes(
                self.N, self.N0, self.zN, self.h[j], self.nmodes)

    def eval_matching_matrices_constant_N(self, h, nmodes):
        """Evaluate the matrices derived from the matching conditions

        Inputs
        ------
        h: 1D array or list
            J+1 elements of scaled depths such that ``max(h) = 1``

        Returns
        -------
        E, F, G, H: lists of 2D arrays
            Lists of J+1 elements, where J of the elements are ``nmodes x nmodes``
            matrices
        """
        # Pre-allocate lists for E, F, G, H
        E, F, G, H = [[[] for _ in self.js] for _ in np.r_[:4]]

        filterwarnings('ignore', '.*invalid value encountered in multiply*.')
        # G and E go from j = 0, .., J-1
        for j in self.js[:-1]:
            if h[j] > h[j+1]:
                E[j] = Q_mn(h[j+1], h[j], nmodes).T
                G[j] = P_mn(h[j], nmodes)
            elif h[j] < h[j+1]:
                E[j] = P_mn(h[j], nmodes)
                G[j] = Q_mn(h[j], h[j+1], nmodes)

        # F and H go from j = 1, .., J
        for j in self.js[1:]:
            if h[j] > h[j-1]:
                F[j] = Q_mn(h[j-1], h[j], nmodes).T
                H[j] = P_mn(h[j], nmodes)
            elif h[j] < h[j-1]:
                F[j] = P_mn(h[j], nmodes)
                H[j] = Q_mn(h[j], h[j-1], nmodes)

        self.E, self.F, self.G, self.H = E, F, G, H

    def eval_matching_matrices_variable_N(self, h, phi_nj, nmodes):
        # Pre-allocate lists for E, F, G, H
        E, F, G, H = [[[] for _ in self.js] for _ in np.r_[:4]]

        filterwarnings('ignore', '.*invalid value encountered in multiply*.')
        # G and E go from j = 0, .., J-1
        for j in self.js[:-1]:
            if h[j] > h[j+1]:
                E[j] = Q_mn_variable_N(h[j+1], h[j], phi_nj[j+1], phi_nj[j]).T
                G[j] = P_mn_variable_N(h[j], phi_nj[j])
            elif h[j] < h[j+1]:
                E[j] = P_mn_variable_N(h[j], phi_nj[j])
                G[j] = Q_mn_variable_N(h[j], h[j+1], phi_nj[j], phi_nj[j+1])

        # F and H go from j = 1, .., J
        for j in self.js[1:]:
            if h[j] > h[j-1]:
                F[j] = Q_mn_variable_N(h[j-1], h[j], phi_nj[j-1], phi_nj[j]).T
                H[j] = P_mn_variable_N(h[j], phi_nj[j])
            elif h[j] < h[j-1]:
                F[j] = P_mn_variable_N(h[j], phi_nj[j])
                H[j] = Q_mn_variable_N(h[j], h[j-1], phi_nj[j], phi_nj[j-1])

        self.E, self.F, self.G, self.H = E, F, G, H

    def update_decay_scales_constant_N(self):
        nvec = np.r_[:self.nmodes]
        gamma_nj = [np.sqrt((nvec*np.pi/hj)**2 + self.lhat**2) for hj in self.h]
        # First and last values of gamma_nj are simply alpha and beta
        self.alpha = gamma_nj[0]
        self.beta = gamma_nj[-1]
        self.gamma_nj = gamma_nj

    def update_decay_scales_variable_N(self):
        gamma_nj = [np.sqrt(gn2_m_lh2_i + self.lhat**2)
                    for gn2_m_lh2_i in self.gn2_m_lh2]
        # First and last values of gamma_nj are simply alpha and beta
        self.alpha = gamma_nj[0]
        self.beta = gamma_nj[-1]
        self.gamma_nj = gamma_nj

    def update_epsilon_zeta(self):
        # epsilon and zeta go from j = 1, .., J-1
        epsilon, zeta = [[[] for _ in self.js] for _ in [1, 2]]
        for j in self.js[1:-1]:
            epsilon[j] = self.gamma_nj[j]*np.tanh(self.gamma_nj[j]*self.Lj_hat[j])
            zeta[j] = self.gamma_nj[j]/np.tanh(self.gamma_nj[j]*self.Lj_hat[j])

        self.epsilon, self.zeta = epsilon, zeta

    def update_block_matrices(self):
        # Unpack names for clarity within this method
        nmodes, J = self.nmodes, self.J
        E, F, G, H = self.E, self.F, self.G, self.H
        alpha, beta = self.alpha, self.beta
        epsilon, zeta = self.epsilon, self.zeta

        if self.is_coastal_wave:
            L_hat_end = 0.5*self.L_endx2/(self.R*self.depth.max())

        LHS = np.matrix(np.zeros((2*J*nmodes, 2*J*nmodes)))
        RHS = np.matrix(np.zeros((2*J*nmodes, 2*J*nmodes)))

        for j in np.r_[:J]:
            if j == 0:
                R_fill = np.zeros((nmodes, (2*J-3)*nmodes))
                LHS_top = np.block([E[0], -F[1], F[1], R_fill])
                LHS_bot = np.block([-G[0], H[1], -H[1], R_fill])

                G_alpha = G[0]*np.matrix(np.diag(alpha))
                H_epsilon = H[1]*np.matrix(np.diag(epsilon[1]))
                H_zeta = H[1]*np.matrix(np.diag(zeta[1]))
                RHS_bot = np.block([-G_alpha, -H_epsilon, H_zeta, R_fill])

            elif j == J-1 and self.is_coastal_wave:
                L_fill = np.zeros((nmodes, (2*J-3)*nmodes))
                tmp_x = (
                    np.exp(-2*beta*L_hat_end) +
                    (self.omega*beta - self.lhat) /
                    (self.omega*beta + self.lhat) *
                    np.exp(2*beta*L_hat_end))
                tmp_x = np.matrix(np.diag(tmp_x))

                LHS_top = np.block([L_fill, E[J-1], E[J-1], -F[J]*tmp_x])
                LHS_bot = np.block([L_fill, -G[J-1], -G[J-1], H[J]*tmp_x])

                G_epsilon = G[J-1]*np.matrix(np.diag(epsilon[J-1]))
                G_zeta = G[J-1]*np.matrix(np.diag(zeta[J-1]))

                tmp_x = beta*(
                    np.exp(-2*beta*L_hat_end) -
                    (self.omega*beta - self.lhat) /
                    (self.omega*beta + self.lhat) *
                    np.exp(2*beta*L_hat_end))

                RHS_bot = np.block([L_fill, -G_epsilon, -G_zeta,
                                    H[J]*np.matrix(np.diag(tmp_x))])

            elif j == J-1:
                L_fill = np.zeros((nmodes, (2*J-3)*nmodes))
                LHS_top = np.block([L_fill, E[J-1], E[J-1], -F[J]])
                LHS_bot = np.block([L_fill, -G[J-1], -G[J-1], H[J]])

                G_epsilon = G[J-1]*np.matrix(np.diag(epsilon[J-1]))
                G_zeta = G[J-1]*np.matrix(np.diag(zeta[J-1]))
                H_beta = H[J]*np.matrix(np.diag(beta))
                RHS_bot = np.block([L_fill, -G_epsilon, -G_zeta, -H_beta])
            else:
                L_fill = np.zeros((nmodes, (2*j-1)*nmodes))
                R_fill = np.zeros((nmodes, (2*J-2*j-3)*nmodes))
                LHS_top = np.block(
                    [L_fill, E[j], E[j], -F[j+1], F[j+1], R_fill])
                LHS_bot = np.block(
                    [L_fill, -G[j], -G[j], H[j+1], -H[j+1], R_fill])

                G_epsilon = G[j]*np.matrix(np.diag(epsilon[j]))
                G_zeta = G[j]*np.matrix(np.diag(zeta[j]))
                H_epsilon = H[j+1]*np.matrix(np.diag(epsilon[j+1]))
                H_zeta = H[j+1]*np.matrix(np.diag(zeta[j+1]))
                RHS_bot = np.block(
                    [L_fill, -G_epsilon, -G_zeta, -H_epsilon, H_zeta, R_fill])

            top_inds = np.s_[j*nmodes:(j+1)*nmodes]
            bot_inds = np.s_[(J+j)*nmodes:(J+j+1)*nmodes]
            LHS[top_inds, :] = LHS_top
            LHS[bot_inds, :] = self.lhat*LHS_bot
            RHS[bot_inds, :] = RHS_bot

        self.LHS, self.RHS = LHS, RHS

    def pick_next_wavenumber(self, niter):
        """Iterate toward solution by fitting curve of omega vs wavelength"""
        if niter == 1:
            # Can fit a line to a single point, so calc a nearby point
            next_l = 0.9*self.l
        else:
            poly_deg = 1 if len(self.omega_ests) <= 3 else 2
            # omega vs wavelength iterates faster than omega vs wavenumber
            # especially for omega close to 1
            pp = np.polyfit(
                self.omega_ests[-3:], 2*np.pi/np.array(self.l_ests[-3:]),
                poly_deg)
            next_l = 2*np.pi/np.polyval(pp, self.omega)

        if next_l < 0:
            # Negative wavenumber is unphysical and occurs when polyfit
            # extrapolates too far from solution. In this case, fitting
            # omega vs wavenumber is necessary
            pp = np.polyfit(
                self.omega_ests[-3:], self.l_ests[-3:], poly_deg)
            next_l = np.polyval(pp, self.omega)

        self.l = next_l

    def calc_wavelength_and_coefficients(self, tol=0.001, niter_max=10):
        """
        tol: float
            | Accuracy of solution
            | Defaults is 0.001 meaning input and output omega values agree to
            | 3 decimal places
        niter_max: int
            Total number of iterations to attempt
        """
        if self.print_progress:
            print('Iterating')
            print('λ (km)      ω')
            print('{0:2.1f}'.format(2*np.pi/self.l/1e3), flush=True)
        self.l = 2*np.pi/self.lambda_guess
        omega_out = 10  # Need really 'incorrect' value to start with
        niter = 0
        # Record all (omega, l) estimates made during iteration
        self.omega_ests = []
        self.l_ests = []

        # Warnings that arise when finding eigenvalue don't affect the
        # low modes, and are consequently unimportant
        filterwarnings('ignore', '.*divide by zero*.')
        filterwarnings(
            'ignore', '.*invalid value encountered in true_divide*.')
        while (abs((self.omega - omega_out)/self.omega) > tol and
               niter < niter_max) or niter < 2:
            niter += 1
            self.update_horizontal_scale()
            if self.variable_N:
                self.update_decay_scales_variable_N()
            else:
                self.update_decay_scales_constant_N()
            self.update_epsilon_zeta()
            self.update_block_matrices()

            omegas = eigvals(self.LHS, self.RHS)
            # Occasionally, the first omega is unphysical
            # either inf, really large, or negative
            # Why? unknown.
            good_eig_solns = np.logical_and(omegas > 0, omegas < 1)
            omegas = omegas[good_eig_solns]

            omega_out = np.real(omegas[self.mode])

            # Record and print iteration
            self.omega_ests += [omega_out]
            self.l_ests += [self.l]
            self.pick_next_wavenumber(niter)
            if self.print_progress:
                # sig. fig. based on tolerance
                sf_tol = -np.floor(np.log10(tol)).astype(int)
                # sig. fig. based on current iteration
                sf_curr = -np.floor(np.log10(
                    np.abs(self.omega - omega_out)/self.omega)).astype(int)
                sf = min(sf_tol, sf_curr)
                fmt_l = '{0:<12.' + str(sf - 1) + 'f}'
                fmt_omega = '{0:<12.' + str(sf + 1) + 'f}'
                try:
                    print(fmt_l.format(2*np.pi/self.l/1e3), end='', flush=True)
                    print(fmt_omega.format(omega_out), flush=True)
                except ValueError:
                    # Initial estimate too far off, don't try printing this step
                    pass

        if self.print_progress:
            if niter < niter_max:
                print('Converged to specified accuracy', flush=True)
            elif niter == niter_max:
                print('Stopping. Number of iterations reached niter_max',
                      flush=True)
            print('Now calculating eigenmode', flush=True)

        omegas, coeffs = eig(self.LHS, self.RHS)
        good_eig_solns = np.logical_and(omegas > 0, omegas < 1)
        omegas = omegas[good_eig_solns]

        coeffs = coeffs[:, good_eig_solns]
        coeffs = np.real(coeffs[:, self.mode]).reshape(2*self.J, self.nmodes)

        self.coeffs = coeffs
        self.wavelength = 2*np.pi/self.l
        self.wavelength_str = (str(np.round(self.wavelength/1e3, decimals=1)) +
                               ' km')
        self.omega = omega_out

        return self.wavelength, self.coeffs

    def calc_mode_shapes(self, x=None):
        """Called by contour_mode_shapes"""
        A = self.coeffs[0]
        B = self.coeffs[-1]
        C = self.coeffs[1:-1:2]
        D = self.coeffs[2:-1:2]
        self.A, self.B, self.C, self.D = A, B, C, D
        C, D = [np.insert(X, 0, np.nan, axis=0) for X in [C, D]]

        if x is None:
            xend = self.Lr/(self.R*self.depth.max())
            if self.is_coastal_wave:
                xend2 = self.L_endx2[0]/(self.R*self.depth.max())
                x = np.r_[self.xhat[0]-xend:self.xhat[-1]+xend2:120j]
            else:
                x = np.r_[self.xhat[0]-xend:self.xhat[-1]+xend:120j]
        else:
            x = np.asanyarray(x, dtype='float')
            x /= self.R*self.depth.max()

        # Ensure each flat region is included in x
        x = np.unique(np.r_[x, midpoint(self.xhat)])

        z = np.r_[-1:0:self.nmodes*2j]
        X, Z = np.meshgrid(x, z, indexing='ij')
        if self.is_coastal_wave:
            L_hat_end = 0.5*self.L_endx2/(self.R*self.depth.max())

        P, U, V = np.zeros_like(X), np.zeros_like(X), np.zeros_like(X)
        mask = np.zeros_like(X).astype(bool)
        nvec = np.r_[:self.nmodes]

        if self.variable_N:
            # Vertically variable part of eigenmode
            def F_z(self, j, niV):
                return self.phi_nj[j][ni](Z)
        else:
            def F_z(self, j, niV):
                return np.cos(ni*np.pi*(Z+self.h[j])/self.h[j])

        for j in self.js:
            if j == 0:
                for ni in nvec:
                    ind = np.s_[x < self.xhat[0], :]

                    P_n = A[ni]*np.exp(self.alpha[ni]*(X-self.xhat[0]))
                    P_n *= F_z(self, 0, ni)
                    dPdx_n = self.alpha[ni]*P_n
                    P[ind] += P_n[ind]
                    U[ind] += (self.omega*dPdx_n - self.lhat*P_n)[ind]
                    V[ind] += (dPdx_n - self.omega*self.lhat*P_n)[ind]
            elif j == self.J and self.is_coastal_wave:
                for ni in nvec:
                    ind = np.s_[x >= self.xhat[-1], :]
                    P_n = B[ni]*np.exp(
                        self.beta[ni]*(X-self.xhat[-1]-2*L_hat_end))
                    P_n += B[ni]*(
                        (self.omega*self.beta[ni] - self.lhat) /
                        (self.omega*self.beta[ni] + self.lhat) *
                        np.exp(-self.beta[ni]*(X-self.xhat[-1]-2*L_hat_end)))
                    P_n *= F_z(self, -1, ni)
                    dPdx_n = B[ni]*self.beta[ni]*np.exp(
                        self.beta[ni]*(X-self.xhat[-1]-2*L_hat_end))
                    dPdx_n -= B[ni]*self.beta[ni]*(
                        (self.omega*self.beta[ni] - self.lhat) /
                        (self.omega*self.beta[ni] + self.lhat) *
                        np.exp(-self.beta[ni]*(X-self.xhat[-1]-2*L_hat_end)))
                    dPdx_n *= F_z(self, -1, ni)
                    P[ind] += P_n[ind]
                    U[ind] += (self.omega*dPdx_n - self.lhat*P_n)[ind]
                    V[ind] += (dPdx_n - self.omega*self.lhat*P_n)[ind]
            elif j == self.J:
                for ni in nvec:
                    ind = np.s_[x >= self.xhat[-1], :]
                    P_n = B[ni]*np.exp(-self.beta[ni]*(X-self.xhat[-1]))
                    P_n *= F_z(self, -1, ni)
                    dPdx_n = -self.beta[ni]*P_n
                    P[ind] += P_n[ind]
                    U[ind] += (self.omega*dPdx_n - self.lhat*P_n)[ind]
                    V[ind] += (dPdx_n - self.omega*self.lhat*P_n)[ind]
            else:
                xj0p5 = (self.xhat[j-1] + self.xhat[j])/2
                ind = np.s_[np.logical_and(
                    x >= self.xhat[j-1], x < self.xhat[j]), :]
                for ni in nvec:
                    gamma = self.gamma_nj[j][ni]
                    X_arg = gamma*(X-xj0p5)
                    C_term = (C[j][ni]*np.cosh(X_arg) /
                              np.cosh(gamma*self.Lj_hat[j]))
                    D_term = (D[j][ni]*np.sinh(X_arg) /
                              np.sinh(gamma*self.Lj_hat[j]))
                    P_n = (C_term + D_term)*F_z(self, j, ni)
                    P[ind] += P_n[ind]
                    dC_dx_term = (C[j][ni]*gamma*np.sinh(X_arg) /
                                  np.cosh(gamma*self.Lj_hat[j]))
                    dD_dx_term = (D[j][ni]*gamma*np.cosh(X_arg) /
                                  np.sinh(gamma*self.Lj_hat[j]))
                    dPdx_n = dC_dx_term + dD_dx_term
                    dPdx_n *= F_z(self, j, ni)
                    U[ind] += (self.omega*dPdx_n - self.lhat*P_n)[ind]
                    V[ind] += (dPdx_n - self.omega*self.lhat*P_n)[ind]

            mask[ind[0][:, None]*(z < -self.h[j]*1.01)[None, :]] = True

            W = np.gradient(P, axis=1)/np.gradient(z)[None, :]

        self.P, self.U, self.V, self.W = P, U, V, W
        self.X, self.Z = X, Z
        self.mask = mask

    def contour_mode_shapes(self, x=None, pcolor=False, Pmax=None, Umax=None,
                            Vmax=None, Wmax=None):
        """Produce contour plots of P, U, V, and W mode shapes

        Inputs
        ------
        x: 1D array (optional)
            | Vector of distances in metres at which mode shapes are evaluated
            | If not provided, x is estimated
        pcolor: bool
            If True, pcolor instead of contour
        Pmax: float
            | vmax or max contour for P axis.
            | Defaults to 1. Reduce if P is washed out
        Umax, Vmax, Wmax: floats
            As for Pmax

        Returns
        -------
        fig: matplotlib fig instance
        axs: the four matplotlib axis objects
        caxs: the contourf or pcolor objects
        """
        try:
            self.calc_mode_shapes(x=x)
        except AttributeError:
            raise AttributeError(
                'calc_wavelength_and_coefficients method must be run')

        Quantities = self.P, self.U, self.V, self.W
        labels = 'PUVW'
        fig, axs = plt.subplots(ncols=2, nrows=2,sharex=True, sharey=True)
        axs[0, 0].set(ylabel='Depth (m)')
        axs[1, 0].set(xlabel='Distance (km)')
        axs = axs.flatten()
        caxs = []

        Pmax = 1 if Pmax is None else Pmax
        Umax = 1 if Umax is None else Umax
        Vmax = 1 if Vmax is None else Vmax
        Wmax = 1 if Wmax is None else Wmax

        for i, Quantity in enumerate(Quantities):
            axi = axs[i]
            X_km = self.X*self.R*self.depth.max()/1e3  # km
            Z_m = self.Z*self.depth.max()
            axi = axs[i]
            vmax = [Pmax, Umax, Vmax, Wmax][i]

            if pcolor:
                caxs += [axi.pcolormesh(
                    X_km, Z_m, ma.masked_where(self.mask, Quantity),
                    cmap='RdBu', vmin=-vmax, vmax=vmax)]
            else:
                caxs += [axi.contourf(
                    X_km, Z_m, ma.masked_where(self.mask, Quantity),
                    cmap='RdBu', levels=np.r_[-vmax:vmax:12j], extend='both')]

            axi.text(0, -self.depth.max()*0.95, labels[i], zorder=3)

            # Add topo
            xin = self.xhat*self.R*self.depth.max()/1e3
            yin = -self.h*self.depth.max()
            xlim = axi.get_xlim()
            if self.is_coastal_wave:
                xin = np.r_[xlim[0], xin,
                            np.ones(2)*(xin[-1]) + self.L_endx2/1e3]
                xin = np.r_[xin, xin[-1] + self.Lr/1e3]
                yin = np.r_[yin, yin[-1], 0, 0]
            else:
                xin = np.r_[xlim[0], xin, xlim[-1]]
                yin = np.r_[yin, yin[-1]]
            axi.fill_between(xin, yin, -self.depth.max(),
                             color='grey', step='post')

        return fig, axs, caxs
