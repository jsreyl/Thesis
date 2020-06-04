import numpy as np
import matplotlib.pyplot as plt

meanFn, phi, zmean, mus, mur = np.loadtxt("meanFn-phi-mus-mur-ALL.txt", unpack=True)

fig, ax = plt.subplots(3, 1, sharex = True, figsize = (12, 12))
ax[0].set_yscale("log")
ax[0].set_xscale("log")
scatter = ax[0].scatter(meanFn, 1.0/phi - 1.32, c=zmean)
ax[0].set_title(r"Colored by $\langle z \rangle$")
colorbar = fig.colorbar(scatter, ax=ax[0])
ax[0].set_ylabel(r"$\langle F_n \rangle$")
ax[0].set_ylim([3.0e-1, 8.0e-1])
ax[0].set_xlim([3.0e-3, 9.0e-2])

scatter = ax[1].scatter(meanFn, 1.0/phi - 1.32, c=mus)
ax[1].set_title(r"Colored by $\mu_s$")
colorbar = fig.colorbar(scatter, ax=ax[1])
ax[1].set_yscale("log")
ax[1].set_ylabel(r"$\langle F_n \rangle$")

scatter = ax[2].scatter(meanFn, 1.0/phi - 1.32, c=mur)
ax[2].set_title(r"Colored by $\mu_r$")
colorbar = fig.colorbar(scatter, ax=ax[2])
ax[2].set_yscale("log")
ax[2].set_xlabel(r"$\frac{1}{\phi} - 1.23$")
ax[2].set_ylabel(r"$\langle F_n \rangle$")


fig.savefig("meanFn-phi.pdf")
fig.savefig("meanFn-phi.png", dpi=300)
