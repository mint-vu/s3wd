import argparse
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import inset_axes, mark_inset

"""
Adapted from Bonet et al. 2023 (https://github.com/clbonet/spherical_sliced-wasserstein)
"""

def compute_stats(data):
    return np.mean(data, axis=0), np.std(data, axis=0)
    
parser = argparse.ArgumentParser()
#parser.add_argument("--prior", type=str, default="unif_sphere", help="Specify prior")
#parser.add_argument("--d_latent", type=int, default=3, help="Dimension of the latent space")
args = parser.parse_args()


L_ess_sw = np.loadtxt("./ess_sw.csv", delimiter=",")
L_ess_ssw = np.loadtxt("./ess_ssw.csv", delimiter=",")
L_kl_sw = np.loadtxt("./kl_sw.csv", delimiter=",")
L_kl_ssw = np.loadtxt("./kl_ssw.csv", delimiter=",")

L_ess_s3w = np.loadtxt("./ess_s3w.csv", delimiter=",")
L_kl_s3w = np.loadtxt("./kl_s3w.csv", delimiter=",")
L_ess_ri1 = np.loadtxt("./ess_ri1.csv", delimiter=",")
L_kl_ri1 = np.loadtxt("./kl_ri1.csv", delimiter=",")
L_ess_ri5 = np.loadtxt("./ess_ri5.csv", delimiter=",")
L_kl_ri5 = np.loadtxt("./kl_ri5.csv", delimiter=",")
L_ess_ri10 = np.loadtxt("./ess_ri10.csv", delimiter=",")
L_kl_ri10 = np.loadtxt("./kl_ri10.csv", delimiter=",")

L_ess_ari30 = np.loadtxt("./ess_ari30.csv", delimiter=",")
L_kl_ari30 = np.loadtxt("./kl_ari30.csv", delimiter=",")
L_ess_ari50 = np.loadtxt("./ess_ari50.csv", delimiter=",")
L_kl_ari50 = np.loadtxt("./kl_ari50.csv", delimiter=",")




# absc = np.array(range(L_ess_sw.shape[1]))*100
# ntry = L_ess_sw.shape[0]

# m_ess_sw = np.mean(np.log10(L_ess_sw), axis=0)
# s_ess_sw = np.std(np.log10(L_ess_sw), axis=0)

# m_ess_sws = np.mean(np.log10(L_ess_sws), axis=0)
# s_ess_sws= np.std(np.log10(L_ess_sws), axis=0)

# m_ess_sw = np.mean(L_ess_sw, axis=0)
# s_ess_sw = np.std(L_ess_sw, axis=0)

# m_ess_sws = np.mean(L_ess_sws, axis=0)
# s_ess_sws= np.std(L_ess_sws, axis=0)

m_ess_sw, s_ess_sw = compute_stats(L_ess_sw)
m_kl_sw, s_kl_sw = compute_stats(L_kl_sw)

m_ess_ssw, s_ess_ssw = compute_stats(L_ess_ssw)
m_kl_ssw, s_kl_ssw = compute_stats(L_kl_ssw)

m_ess_s3w, s_ess_s3w = compute_stats(L_ess_s3w)
m_kl_s3w, s_kl_s3w = compute_stats(L_kl_s3w)

m_ess_ri1, s_ess_ri1 = compute_stats(L_ess_ri1)
m_kl_ri1, s_kl_ri1 = compute_stats(L_kl_ri1)

m_ess_ri5, s_ess_ri5 = compute_stats(L_ess_ri5)
m_kl_ri5, s_kl_ri5 = compute_stats(L_kl_ri5)

m_ess_ri10, s_ess_ri10 = compute_stats(L_ess_ri10)
m_kl_ri10, s_kl_ri10 = compute_stats(L_kl_ri10)

m_ess_ari30, s_ess_ari30 = compute_stats(L_ess_ari30)
m_kl_ari30, s_kl_ari30 = compute_stats(L_kl_ari30)

m_ess_ari50, s_ess_ari50 = compute_stats(L_ess_ari50)
m_kl_ari50, s_kl_ari50 = compute_stats(L_kl_ari50)

absc = np.array(range(L_ess_sw.shape[1])) * 100
ntry = L_ess_sw.shape[0]

# fig = plt.figure(figsize=(6,3))
# plt.plot(absc, m_ess_sws, label="SSWVI")
# plt.fill_between(absc,m_ess_sws-2*s_ess_sws/np.sqrt(ntry),m_ess_sws+2*s_ess_sws/np.sqrt(ntry),alpha=0.5)
# plt.plot(absc, m_ess_sw, label="SWVI")
# plt.fill_between(absc,m_ess_sw-2*s_ess_sw/np.sqrt(ntry),m_ess_sw+2*s_ess_sw/np.sqrt(ntry),alpha=0.5)
# plt.grid(True)
# plt.xlabel("Iterations",fontsize=13)
# plt.title(r"ESS")
# plt.legend(fontsize=13)
# # plt.savefig("./ESS.png", format="png")
# plt.savefig("./ESS.pdf", format="pdf", bbox_inches="tight")
# plt.close("all")




# absc = np.array(range(L_kl_sw.shape[1]))*100
# ntry = L_kl_sw.shape[0]

# m_kl_sw = np.mean(L_kl_sw, axis=0)
# s_kl_sw = np.std(L_kl_sw, axis=0)

# m_kl_sws = np.mean(L_kl_sws, axis=0)
# s_kl_sws= np.std(L_kl_sws, axis=0)


# fig = plt.figure(figsize=(6,3))
# plt.plot(absc, m_kl_sws, label="SSWVI")
# plt.fill_between(absc,m_kl_sws-2*s_kl_sws/np.sqrt(ntry),m_kl_sws+2*s_kl_sws/np.sqrt(ntry),alpha=0.5)
# plt.plot(absc, m_kl_sw, label="SWVI")
# plt.fill_between(absc,m_kl_sw-2*s_kl_sw/np.sqrt(ntry),m_kl_sw+2*s_kl_sw/np.sqrt(ntry),alpha=0.5)
# plt.grid(True)
# plt.xlabel("Iterations",fontsize=13)
# plt.title(r"KL")
# plt.legend(fontsize=13)
# plt.savefig("./KL.png", format="png")
# plt.savefig("./KL.pdf", format="pdf", bbox_inches="tight")
# plt.close("all")
##################################################################

fig = plt.figure(figsize=(6,3))
plt.plot(absc, m_ess_sw, label="SW-VI")
plt.fill_between(absc, m_ess_sw-2*s_ess_sw/np.sqrt(ntry), m_ess_sw+2*s_ess_sw/np.sqrt(ntry), alpha=0.5)

# Add plots for new distances
plt.plot(absc, m_ess_s3w, label="S3W-VI")
plt.fill_between(absc, m_ess_s3w-2*s_ess_s3w/np.sqrt(ntry), m_ess_s3w+2*s_ess_s3w/np.sqrt(ntry), alpha=0.5)
plt.plot(absc, m_ess_ri1, label="RIS3W-VI (1)")
plt.fill_between(absc, m_ess_ri1-2*s_ess_ri1/np.sqrt(ntry), m_ess_ri1+2*s_ess_ri1/np.sqrt(ntry), alpha=0.5)
plt.plot(absc, m_ess_ri5, label="RIS3W-VI (5)")
plt.fill_between(absc, m_ess_ri5-2*s_ess_ri5/np.sqrt(ntry), m_ess_ri5+2*s_ess_ri5/np.sqrt(ntry), alpha=0.5)
plt.plot(absc, m_ess_ri10, label="RIS3W-VI (10)")
plt.fill_between(absc, m_ess_ri10-2*s_ess_ri10/np.sqrt(ntry), m_ess_ri10+2*s_ess_ri10/np.sqrt(ntry), alpha=0.5)
plt.plot(absc, m_ess_ari30, label="ARIS3W-VI (30/1000)")
plt.fill_between(absc, m_ess_ari30-2*s_ess_ari30/np.sqrt(ntry), m_ess_ari30+2*s_ess_ari30/np.sqrt(ntry), alpha=0.5)
plt.plot(absc, m_ess_ari50, label="ARIS3W-VI (50/1000)")
plt.fill_between(absc, m_ess_ari50-2*s_ess_ari50/np.sqrt(ntry), m_ess_ari50+2*s_ess_ari50/np.sqrt(ntry), alpha=0.5)
plt.plot(absc, m_ess_ssw, label="SSW-VI")
plt.fill_between(absc, m_ess_ssw-2*s_ess_ssw/np.sqrt(ntry), m_ess_ssw+2*s_ess_ssw/np.sqrt(ntry), alpha=1)
# plt.set_xscale('log')
# plt.yscale('log')

plt.grid(True)
plt.xlabel("Iterations", fontsize=10)
plt.ylabel("ESS", fontsize=10)
plt.title("Effective Sample Size (ESS) Over Iterations")
plt.legend(fontsize=10)
plt.savefig("./ESS.pdf", format="pdf", bbox_inches="tight")

plt.close(fig)

absc = np.array(range(L_kl_sw.shape[1])) * 100
ntry = L_kl_sw.shape[0]

fig = plt.figure(figsize=(6,3))
plt.plot(absc, m_kl_sw, label="SW-VI")
plt.fill_between(absc, m_kl_sw-2*s_kl_sw/np.sqrt(ntry), m_kl_sw+2*s_kl_sw/np.sqrt(ntry), alpha=0.5)


plt.plot(absc, m_kl_s3w, label="S3W-VI")
plt.fill_between(absc, m_kl_s3w-2*s_kl_s3w/np.sqrt(ntry), m_kl_s3w+2*s_kl_s3w/np.sqrt(ntry), alpha=0.5)
plt.plot(absc, m_kl_ri1, label="RIS3W-VI (1)")
plt.fill_between(absc, m_kl_ri1-2*s_kl_ri1/np.sqrt(ntry), m_kl_ri1+2*s_kl_ri1/np.sqrt(ntry), alpha=0.5)
plt.plot(absc, m_kl_ri5, label="RIS3W-VI (5)")
plt.fill_between(absc, m_kl_ri5-2*s_kl_ri5/np.sqrt(ntry), m_kl_ri5+2*s_kl_ri5/np.sqrt(ntry), alpha=0.5)
plt.plot(absc, m_kl_ri10, label="RIS3W-VI (10)")
plt.fill_between(absc, m_kl_ri10-2*s_kl_ri10/np.sqrt(ntry), m_kl_ri10+2*s_kl_ri10/np.sqrt(ntry), alpha=0.5)
plt.plot(absc, m_kl_ari30, label="ARIS3W-VI (30/1000)")
plt.fill_between(absc, m_kl_ari30-2*s_kl_ari30/np.sqrt(ntry), m_kl_ari30+2*s_kl_ari30/np.sqrt(ntry), alpha=0.5)
plt.plot(absc, m_kl_ari50, label="ARIS3W-VI (50/1000)")
plt.fill_between(absc, m_kl_ari50-2*s_kl_ari50/np.sqrt(ntry), m_kl_ari50+2*s_kl_ari50/np.sqrt(ntry), alpha=0.5)
# plt.yscale('log')
plt.plot(absc, m_kl_ssw, label="SSW-VI")
plt.fill_between(absc, m_kl_ssw-2*s_kl_ssw/np.sqrt(ntry), m_kl_ssw+2*s_kl_ssw/np.sqrt(ntry), alpha=1)
plt.grid(True)
plt.xlabel("Iterations", fontsize=10)
plt.ylabel("KL Divergence", fontsize=10)
plt.title("KL Divergence Over Iterations")
plt.legend(fontsize=10)
plt.savefig("./KL.pdf", format="pdf", bbox_inches="tight")
plt.close(fig)


















