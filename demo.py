import numpy as np
import scipy.signal as signal


def remezBands(L, w0):
    """Band edges and gains of an interpolation filter for use with Remez.

    Parameters
    ----------
    L : int
        Interpolation factor, e.g., 2, 3, etc.
    w0 : float
        Signal bandwidth in radians, i.e., 0 < `w0` < π.

    Returns
    -------
    bands : ndarray
        A rank-1 array (vector) of band edges in normalized frequency. All
         entries are between 0 and 0.5.
    gains : ndarray
        Vector half-as-long as `bands` containing gains in these bands. These
        two can be passed into, e.g., `scipy.signal.remez`.

    References
    ----------
    .. [1] Phil Schniter, “ECE-700 Multirate Notes”, March 27, 2006, page 6.
           http://www2.ece.ohio-state.edu/~schniter/ee700/handouts/multirate.pdf
    """
    if L % 2 == 0:
        k = np.arange(0, L / 2.0)
        transitionBands = np.sort(np.hstack(
            (0, np.pi, (2 * k * np.pi + w0) / L, (2 * (k + 1) * np.pi - w0) / L
             ))) / (2 * np.pi)
    else:
        k = np.arange(0, (L + 1) / 2.0)
        transitionBands = np.sort(np.hstack(
            (0, (2 * k * np.pi + w0) / L, (2 * (k + 1) * np.pi - w0) / L
             ))) / (2 * np.pi)
        transitionBands = transitionBands[:-1]
    desired = np.zeros(transitionBands.size // 2)
    desired[0] = 1.0
    return transitionBands, desired


def remezDesign(L, w0, rippleDb=-30):
    """Design an interpolation filter with Remez exchange.

    Parameters
    ----------
    L : int
        Interpolation factor, e.g., 2, 3, etc.
    w0 : float
        Bandwidth of the signal in radians, i.e., 0 < `w0` < π.
    rippleDb : float
        Tolerable ripple (in 10*log10 dB). This is used with Kaiser’s formula
        to estimate the filter length. E.g., ripple (max in passband, min in
        stopband) of 0.001 means `rippledB` of -30.

    Returns
    -------
    out : ndarray
        A rank-1 array (vector) containing the filter weights.
    """
    ntaps, _ = signal.kaiserord(rippleDb,
                                (2 * np.pi - 2 * w0) / (L * 2 * np.pi))
    bandsDesired = remezBands(L, w0)
    return signal.remez(ntaps, *bandsDesired)


def firwin2Design(L, w0, rippleDb=-30):
    """Like `remezDesign` but uses `scipy.signal.firwin2` instead of Remez.

    Since `firwin2` doesn’t handle don’t-care regions, the returned filter is
    just an ordinary low-pass filter, and is likely to have slower
    transition-band behavior than that returned by `remezDesign` or
    `firlsDesign`.
    """
    ntaps, _ = signal.kaiserord(rippleDb,
                                (2 * np.pi - 2 * w0) / (L * 2 * np.pi))
    bands, gains = remezBands(L, w0)
    return signal.firwin2(ntaps, [0, bands[1], bands[2], 0.5], [1, 1, 0, 0.0],
                          nyq=0.5)


def firlsDesign(L, w0, rippleDb=-30):
    "Like `remezDesign` but uses `scipy.signal.firls` instead of Remez."
    ntaps, _ = signal.kaiserord(rippleDb,
                                (2 * np.pi - 2 * w0) / (L * 2 * np.pi))
    # firls requires odd numtaps
    if ntaps % 2 == 0:
        ntaps += 1
    b, g = remezBands(L, w0)
    # the `vstack` stuff below: repeat each element of `g` twice, so `[1 0 0]`
    # becomes `[1 1, 0 0, 0 0]`.
    return signal.firls(ntaps, b, np.vstack([g, g]).T.ravel(), nyq=0.5)


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    db20 = lambda x: 20 * np.log10(np.abs(x))

    def viz(b):
        w, h = signal.freqz(b)
        plt.plot(w / (2 * np.pi), db20(h))

    w0 = 0.9 * np.pi

    Ntaps = 96
    plt.figure()
    viz(signal.remez(Ntaps, *remezBands(5, w0)))
    viz(signal.remez(Ntaps, *remezBands(4, w0)))
    viz(signal.remez(Ntaps, *remezBands(3, w0)))
    viz(signal.remez(Ntaps, *remezBands(2, w0)))
    plt.grid()

    plt.figure()
    viz(remezDesign(5, w0))
    viz(remezDesign(4, w0))
    viz(remezDesign(3, w0))
    viz(remezDesign(2, w0))
    plt.grid()

    plt.figure()
    viz(remezDesign(4, w0))
    viz(firwin2Design(4, w0))
    viz(firlsDesign(4, w0))
    plt.grid()
