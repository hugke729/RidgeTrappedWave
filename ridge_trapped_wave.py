import sys
from warnings import filterwarnings, catch_warnings, warn
import numpy as np
import numpy.ma as ma
from scipy.linalg import eig, eigvals
import matplotlib.pyplot as plt


def remove_duplicate_depths(x, depths):
    dup_idx = np.isclose(depths[:-1], depths[1:])
    if any(dup_idx):
        warn_msg = (str(sum(dup_idx)) + ' adjacent duplicate depths in input'
                    ' have been removed\n')
        warn(warn_msg)
    x = x[~dup_idx]
    depths = depths[~np.append(dup_idx, False)]
    return x, depths


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
        denominator = h1**2*m**2*np.pi-h2**2*n**2*pi
        Q_mn = np.matrix(numerator/denominator)

        # Deal with singularity
        sing = np.isclose(m*h1, n*h2)
        Q_mn[sing] = ((-1)**n/2*(h1*np.cos(h2*n*pi/h1)))[sing]

        Q_mn[0, 0] = h1
    return Q_mn


class RidgeTrappedWave(object):
    """Calculate properties of an subinertial wave trapped along a ridge

    See Hughes et al. (2018) Tidal conversion and dissipation at steep
    topography in a channel poleward of the critical latitude. In prep for
    J. Phys. Oceanogr.

    Specifically, iteratively solve an eigenvalue problem that arises by
    matching pressure and velocity at each step discontinuity.

    omega is the eigenvalue of the problem, but omega is also used to
    construct the LHS and RHS matrices. Hence, it's actually a non-linear
    problem. This is overcome by solving iteratively.

    It is assumed that omega is known, and wavenumber `l` is unknown,
    so in each iteration `l` is adjusted

    Inputs
    ------
    x: 1D array
        Array defining the locations of the discontinuites (metres)
    depths: 1D array
        Array defining the depths (metres). This must be one element larger
        than ``x``
    omega: float
        | Non-dimensionalised tidal forcing frequency
        | Note that omega is dimensional frequency divided by coriolis
        frequency so that 0 < omega < 1
    N0: float
        Buoyancy frequency (rad/s)
    lat: float
        | Latitude in decimal degrees (needed for Coriolis frequency)
        | May not work for negative latitudes
    lambda_guess: float (optional)
        First estimate of the wavelength (m)
    nmodes: int
        Number of vertical modes (default of 20).
    mode: int
        | Which mode to solve for. Defaults to lowest mode (0)
        | Python counts from 0, so ``mode = 0`` is the lowest baroclinic mode
        | ``mode >= 1`` haven't been thoroughly tested. Proceed with caution

    Example
    -------
    >>> x = 0, 3000
    >>> z = 250, 50, 250
    >>> omega = 0.9
    >>> N0 = 6E-3
    >>> lat = 80
    >>> r = RidgeTrappedWave(x, z, omega, N0, lat, lambda_guess=40e3)
    >>> r.calc_wavelength_and_coefficients(niter_max=8, print_iterations=True)
    >>> # r.contour_mode_shapes() # Plot solution
    >>> print(r)
    λ (km)      ω
    40.0
    44.44       0.92
    46.116      0.905
    46.133      0.900
    Converged to specified accuracy
    Ridge Trapped Wave with wavelength of 46.1 km
    """

    def __init__(
            self, x, depth, omega, N0, lat, lambda_guess=None,
            nmodes=20, mode=0):

        # Check and adjust inputs before doing anything
        x = np.asanyarray(x)
        depth = np.asanyarray(depth)
        assert x.size >= 2, ('This code does not work for a single step. '
                             'Perhaps try faking it with two very closely '
                             'spaced steps')
        assert len(depth) == len(x) + 1, 'len(depths) must equal len(x) + 1'
        assert all(depth > 0), 'Depths must be positive'
        assert 0 < omega < 1, '0 < omega < 1 (omega is non-dimensionalised by f)'
        assert np.all(np.diff(x) > 0), 'x must increase monotonically'
        if len(x)*nmodes > 500:
            M = str(2*len(x)*nmodes)
            warn('Matrices are large (' + M + ' x ' + M + ')\n'
                 'Computation may take several minutes')
            sys.stderr.flush()

        x, depth = remove_duplicate_depths(x, depth)

        self.x = x
        self.depth = depth
        self.omega = omega
        self.N0 = N0
        self.lat = lat
        self.h = depth/depth.max()
        self.lat_to_f(lat)
        if lambda_guess is None:
            lambda_guess = (2*N0*depth.max()/(omega*self.f))/(1-self.h.min())
            self.lambda_guess = lambda_guess/(mode+1)
        else:
            self.lambda_guess = lambda_guess
        self.l = 2*np.pi/self.lambda_guess
        self.nmodes = nmodes
        self.mode = mode

        self.jvec()
        self.update_R()
        self.update_horizontal_scale()
        self.eval_matching_matrices(self.h, self.nmodes)
        self.wavelength_str = ('UNDETERMINED. Run calc_wavelength_and_coefficients '
                               'method to determine')

    def __repr__(self):
        x_str = '[' + ', '.join(str(np.round(
            xi, decimals=1)) for xi in self.x) + ']'
        depth_str = '[' + ', '.join(str(np.round(
            zi, decimals=1)) for zi in self.depth) + ']'
        fmt_str = ('{}(omega={!s}, N0={!s}, lat={!s}, lambda_guess={!s}, '
                   'nmodes={!s}, mode={!s},\nx={},\ndepth={!r})')
        return fmt_str.format(
            self.__class__.__name__,
            self.omega, self.N0, self.lat, self.lambda_guess, self.nmodes,
            self.mode, x_str, depth_str)

    def __str__(self):
        return 'Ridge Trapped Wave with wavelength of ' + self.wavelength_str

    def lat_to_f(self, lat):
        self.f = 2*7.29211E-5*np.sin(np.deg2rad(lat))

    def jvec(self):
        """Total number of discontinuties and helpful counter"""
        self.J = len(self.x)
        self.js = np.r_[:self.J+1]

    def update_R(self):
        """R is defined by Chapman (1982) doi:10.1016/0377-0265(82)90002-1"""
        # self.R = (self.N0/self.f)/np.sqrt(1-self.omega**2)
        # See equation 4 for unapproximated R
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

    def eval_matching_matrices(self, h, nmodes):
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
        # Pre-allocate lists for E, F, G, H, epsilon, and zeta
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

    def update_decay_scales(self):
        nvec = np.r_[:self.nmodes]
        gamma_nj = [np.sqrt((nvec*np.pi/hj)**2 + self.lhat**2) for hj in self.h]
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

    def pick_next_wavenumber(self, niter, print_iterations):
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

    def calc_wavelength_and_coefficients(
            self, tol=0.001, niter_max=10, print_iterations=True):
        """
        tol: float
            | Accuracy of solution
            | Defaults is 0.001 meaning input and output omega values agree to
            | 3 decimal places
        niter_max: int
            Total number of iterations to attempt
        print_iterations: bool
            Set to False to suppress output regarding iterations
        """
        if print_iterations:
            print('λ (km)      ω')
            print('{0:2.1f}'.format(2*np.pi/self.l/1e3), flush=True)
        self.l = 2*pi/self.lambda_guess
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
               niter < niter_max):
            niter += 1
            self.update_horizontal_scale()
            self.update_decay_scales()
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
            self.pick_next_wavenumber(niter, print_iterations)
            if print_iterations:
                # sig. fig. based on tolerance
                sf_tol = -np.floor(np.log10(tol)).astype(int)
                # sig. fig. based on current iteration
                sf_curr = -np.floor(np.log10(
                    np.abs(self.omega - omega_out)/self.omega)).astype(int)
                sf = min(sf_tol, sf_curr)
                fmt = '{0:<12.' + str(sf) + 'f}'
                print(fmt.format(2*np.pi/self.l/1e3), end='', flush=True)
                print(fmt.format(omega_out), flush=True)

        if print_iterations:
            if niter < niter_max:
                print('Converged to specified accuracy', flush=True)
            elif niter == niter_max:
                print('Stopping. Number of iterations reached niter_max',
                      flush=True)
            print('Now calculating eigenvectors', flush=True)

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
            x = np.r_[self.xhat[0] - 1:self.xhat[-1] + 1:120j]
        else:
            x = np.asanyarray(x, dtype='float')
            x /= self.R*self.depth.max()

        z = np.r_[-1:0:self.nmodes*2j]
        X, Z = np.meshgrid(x, z, indexing='ij')

        P, U, V = np.zeros_like(X), np.zeros_like(X), np.zeros_like(X)
        mask = np.zeros_like(X).astype(bool)
        nvec = np.r_[:self.nmodes]

        for j in self.js:
            if j == 0:
                for ni in nvec:
                    ind = np.s_[x < self.xhat[0], :]
                    P_n = (A[ni]*np.exp(self.alpha[ni]*(X-self.xhat[0])) *
                           np.cos(ni*np.pi*(Z+self.h[0])/self.h[0]))
                    P[ind] += P_n[ind]

                    dPdx_n = self.alpha[ni]*P_n
                    U[ind] += (self.omega*dPdx_n - self.lhat*P_n)[ind]
                    V[ind] += (dPdx_n - self.omega*self.lhat*P_n)[ind]
            elif j == self.J:
                for ni in nvec:
                    ind = np.s_[x >= self.xhat[-1], :]
                    P_n = (B[ni]*np.exp(-self.beta[ni]*(X-self.xhat[-1])) *
                           cos(ni*np.pi*(Z+self.h[-1])/self.h[-1]))
                    P[ind] += P_n[ind]

                    dPdx_n = -self.beta[ni]*P_n
                    U[ind] += (self.omega*dPdx_n - self.lhat*P_n)[ind]
                    V[ind] += (dPdx_n - self.omega*self.lhat*P_n)[ind]
            else:
                for ni in nvec:
                    gamma = self.gamma_nj[j][ni]
                    xj0p5 = (self.xhat[j-1] + self.xhat[j])/2
                    ind = np.s_[np.logical_and(
                        x >= self.xhat[j-1], x < self.xhat[j]), :]
                    X_arg = gamma*(X-xj0p5)

                    C_term = (C[j][ni]*np.cosh(X_arg) /
                              np.cosh(gamma*self.Lj_hat[j]))
                    D_term = (D[j][ni]*np.sinh(X_arg) /
                              np.sinh(gamma*self.Lj_hat[j]))
                    P_n = ((C_term + D_term) *
                           np.cos(ni*np.pi*(Z+self.h[j])/self.h[j]))
                    P[ind] += P_n[ind]

                    dC_dx_term = (C[j][ni]*gamma*np.sinh(X_arg) /
                                  np.cosh(gamma*self.Lj_hat[j]))
                    dD_dx_term = (D[j][ni]*gamma*np.cosh(X_arg) /
                                  np.sinh(gamma*self.Lj_hat[j]))
                    dPdx_n = ((dC_dx_term + dD_dx_term) *
                              np.cos(ni*np.pi*(Z+self.h[j])/self.h[j]))
                    U[ind] += (self.omega*dPdx_n - self.lhat*P_n)[ind]
                    V[ind] += (dPdx_n - self.omega*self.lhat*P_n)[ind]

            mask[ind[0][:, None]*(z < -self.h[j])[None, :]] = True

            W = np.gradient(P, axis=1)/np.gradient(z)[None, :]

        self.P, self.U, self.V, self.W = P, U, V, W
        self.X, self.Z = X, Z
        self.mask = mask

    def contour_mode_shapes(self, x=None, pcolor=False):
        """
        Produce contour plots of P, U, V, and W mode shapes

        Inputs
        ------
        x: 1D array (optional)
            | Vector of distances in metres at which mode shapes are evaluated
            | If not provided, x is guessed, sometimes poorly
        pcolor: bool
            If True, pcolor instead of contour

        Returns
        -------
        fig: matplotlib fig instance
        axs: the four matplotlib axis objects
        caxs: the contourf objects
        """
        self.calc_mode_shapes(x=x)

        Quantities = self.P, self.U, self.V, self.W
        labels = 'PUVW'
        fig, axs = plt.subplots(ncols=2, nrows=2,sharex=True, sharey=True)
        axs[0, 0].set(ylabel='Depth (m)')
        axs[1, 0].set(xlabel='Distance (km)')
        axs = axs.flatten()
        vmax = 1
        caxs = []
        for i, Quantity in enumerate(Quantities):
            axi = axs[i]
            X_km = self.X*self.R*self.depth.max()/1e3  # km
            Z_m = self.Z*self.depth.max()
            axi = axs[i]

            if pcolor:
                caxs += [axi.pcolormesh(
                    X_km, Z_m, ma.masked_where(self.mask, Quantity),
                    cmap=rygcb(), vmin=-vmax, vmax=vmax)]
            else:
                caxs += [axi.contourf(
                    X_km, Z_m, ma.masked_where(self.mask, Quantity),
                    cmap=rygcb(), levels=np.r_[-vmax:vmax:12j], extend='both')]

            axi.text(0, -self.depth.max()*0.95, labels[i], zorder=3)

            # Add topo
            xin = self.xhat*self.R*self.depth.max()/1e3
            yin = -self.h*self.depth.max()
            xlim = axi.get_xlim()
            xin = np.r_[xlim[0], xin, xlim[-1]]
            yin = np.r_[yin, yin[-1]]
            axi.fill_between(xin, yin, -self.depth.max(),
                             color='grey', step='post')

        return fig, axs, caxs
