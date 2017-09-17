import numpy as np
import scipy.signal as signal


def remezBands(L, w0):
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


def viz(b):
    import matplotlib.pyplot as plt
    db20 = lambda x: 20 * np.log10(np.abs(x))
    w, h = signal.freqz(b)
    plt.plot(w / (2 * np.pi), db20(h))


def remezDesign(L, w0, rippleDb=-30):
    bandsDesired = remezBands(L, w0)
    ntaps, _ = signal.kaiserord(rippleDb,
                                (2 * np.pi - 2 * w0) / (L * 2 * np.pi))
    return signal.remez(ntaps, *bandsDesired)


w0 = 0.9 * np.pi

import matplotlib.pyplot as plt

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
