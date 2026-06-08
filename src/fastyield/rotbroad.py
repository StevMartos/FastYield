import numpy as np

C_KMS = 299792.458

# Taken from PyAstronomy: https://pyastronomy.readthedocs.io/en/latest/index.html



class PyAValError(ValueError):
    """
    Minimal replacement for PyAstronomy.pyaC.pyaErrors.PyAValError.
    Keeps a similar signature (message, where=..., solution=...).
    """
    def __init__(self, message, where=None, solution=None):
        super().__init__(message)
        self.where = where
        self.solution = solution

    def __str__(self):
        s = super().__str__()
        if self.where is not None:
            s += f" (where={self.where})"
        if self.solution is not None:
            s += f" (solution={self.solution})"
        return s


class _Gdl:
    def __init__(self, vsini, epsilon):
        """
        Broadening profile (Gray) with linear limb darkening.

        vsini   : km/s
        epsilon : linear limb-darkening coefficient in [0, 1]
        """
        self.vc = float(vsini) / C_KMS
        self.eps = float(epsilon)

    def gdl(self, dl, refwvl, dwl):
        """
        dl     : delta-lambda array (same unit as refwvl, typically Angstrom)
        refwvl : reference wavelength (scalar)
        dwl    : wavelength bin size
        """
        dl = np.asarray(dl, dtype=float)
        refwvl = float(refwvl)
        dwl = float(dwl)

        dlmax = self.vc * refwvl
        if dlmax <= 0:
            return np.zeros_like(dl)

        c1 = 2.0 * (1.0 - self.eps) / (np.pi * dlmax * (1.0 - self.eps / 3.0))
        c2 = self.eps / (2.0 * dlmax * (1.0 - self.eps / 3.0))

        x            = dl / dlmax
        result       = np.zeros_like(x)
        indi         = np.where(np.abs(x) < 1.0)[0]
        one_minus_x2 = 1.0 - x[indi] ** 2
        result[indi] = c1 * np.sqrt(one_minus_x2) + c2 * one_minus_x2

        # Discrete renormalization (same idea as PyAstronomy)
        area = np.sum(result) * dwl
        if area > 0:
            result /= area
        return result


def _check_inputs(wvl, flux, epsilon, vsini, where):
    wvl = np.asarray(wvl, dtype=float)
    flux = np.asarray(flux, dtype=float)

    if wvl.ndim != 1 or flux.ndim != 1:
        raise PyAValError("wvl and flux must be 1D arrays.", where=where)
    if wvl.size != flux.size:
        raise PyAValError("wvl and flux must have the same length.", where=where)
    if wvl.size < 2:
        raise PyAValError("wvl/flux must contain at least two points.", where=where)

    sp = np.diff(wvl)
    if not np.allclose(sp, sp[0], rtol=0.0, atol=1e-6):
        raise PyAValError(
            "Input wavelength array is not evenly spaced.",
            where=where,
            solution=[
                "Use evenly spaced input array.",
                "Interpolate to an equidistant grid first."
            ],
        )

    vsini = float(vsini)
    epsilon = float(epsilon)

    if vsini <= 0.0:
        raise PyAValError("vsini must be positive.", where=where)
    if (epsilon < 0.0) or (epsilon > 1.0):
        raise PyAValError(
            "Linear limb-darkening coefficient epsilon must satisfy 0 <= epsilon <= 1.",
            where=where,
            solution="Adapt epsilon."
        )

    return wvl, flux, epsilon, vsini


def rotBroad(wvl, flux, epsilon, vsini, edgeHandling="firstlast"):
    """
    Rotational broadening (Gray) with wavelength-dependent kernel.
    Requires an evenly spaced wavelength grid.

    edgeHandling:
      - "firstlast": pad using first/last flux value (recommended)
      - "None": no padding, convolution edge effects remain
    """
    wvl, flux, epsilon, vsini = _check_inputs(wvl, flux, epsilon, vsini, where="rotBroad")
    dwl = wvl[1] - wvl[0]

    if edgeHandling == "firstlast":
        binnu = int(np.floor(((vsini / C_KMS) * np.max(wvl)) / dwl)) + 1

        validIndices = np.arange(flux.size) + binnu

        flux_pad  = np.concatenate([np.full(binnu, flux[0]), flux, np.full(binnu, flux[-1])])
        wvl_front = (wvl[0] - (np.arange(binnu) + 1) * dwl)[::-1]
        wvl_end   = wvl[-1] + (np.arange(binnu) + 1) * dwl
        wvl_pad   = np.concatenate([wvl_front, wvl, wvl_end])

        flux = flux_pad
        wvl  = wvl_pad

    elif edgeHandling == "None":
        validIndices = np.arange(flux.size)
    else:
        raise PyAValError(
            f"Edge handling method '{edgeHandling}' not supported.",
            where="rotBroad",
            solution="Choose 'firstlast' or 'None'."
        )

    gdl = _Gdl(vsini, epsilon)
    result = np.zeros_like(flux, dtype=float)

    for i in range(flux.size):
        dl = wvl[i] - wvl
        g = gdl.gdl(dl, wvl[i], dwl)
        result[i] = np.sum(flux * g)

    # Keep same convention as PyAstronomy
    result *= dwl
    return result[validIndices]


def fastRotBroad(wvl, flux, epsilon, vsini, effWvl=None):
    """
    Faster rotational broadening using a single kernel evaluated at effWvl.
    Requires an evenly spaced wavelength grid. Uses np.convolve(mode="same")
    so edge effects remain (by design).
    """
    wvl, flux, epsilon, vsini = _check_inputs(wvl, flux, epsilon, vsini, where="fastRotBroad")
    dwl = wvl[1] - wvl[0]

    if effWvl is None:
        effWvl = float(np.mean(wvl))
    else:
        effWvl = float(effWvl)

    gdl = _Gdl(vsini, epsilon)

    binnHalf = int(np.floor(((vsini / C_KMS) * effWvl / dwl))) + 1
    gwvl     = (np.arange(4 * binnHalf) - 2 * binnHalf) * dwl + effWvl

    dl = gwvl - effWvl
    g = gdl.gdl(dl, effWvl, dwl)

    # Remove zeros
    g = g[g > 0.0]

    return np.convolve(flux, g, mode="same") * dwl
