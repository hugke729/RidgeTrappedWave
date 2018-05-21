# RidgeTrappedWave
Python code to calculate properties of a Kelvin wave trapped at a ridge

This is a supplement to a paper in preparation:

Hughes and Klymak (2018) Tidal conversion and dissipation at steep topography in a channel poleward of the critical latitude

Using this code without the associated paper is not recommended. Please email me if interested at this early stage.

As described in the docstring, ridge-trapped wave properties are calculated following the example below

```python
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
```

