import sys
import numpy as np
import matplotlib.pyplot as plt
from colorama import Fore, Back, Style
from scipy.stats import gamma
from scipy.optimize import curve_fit

INFO=Fore.GREEN+'# INFO: '+Style.RESET_ALL
ERROR=Fore.RED+'# ERROR: '+Style.RESET_ALL
WARNING=Fore.YELLOW+'# WARNING: '+Style.RESET_ALL

class INFOPRINT:
    def __init__(self, message):
        self.message_ = message
        print(INFO+"START -> "+self.message_)
    def __del__(self):
        print(INFO+"DONE -> "+self.message_+"\n")

def read_histogram(fname):
    auxlog=INFOPRINT(f"Reading pressure histogram from file {fname}")
    pressBins, nums = np.loadtxt(fname, unpack=True)
    return pressBins, nums
def read_pressure(fname):
    auxlog=INFOPRINT(f"Reading pressure per particle from file {fname}")
    press = np.loadtxt(fname, unpack=True)
    return press

def read_cundall(fname):
    auxlog=INFOPRINT("Reading cundall data")
    timestep, cundall_filtered, cundall_full = np.loadtxt(fname, unpack=True)
    return timestep, cundall_filtered, cundall_full

def read_mean_coordination_number(fname):
    auxlog=INFOPRINT("Reading mean coordination number data")
    timestep, z = np.loadtxt(fname, unpack=True)
    return timestep, z

def read_packing_fraction(fname):
    auxlog=INFOPRINT(f"Reading packing fraction data from file {fname}")
    timestep, packing_fraction = np.loadtxt(fname, unpack=True)
    return timestep, packing_fraction

def k_gamma(x,k,theta):
    return gamma.pdf(x,a=k,scale=theta)

################################################
############ MAIN IMPLEMENTATION ###############
################################################

#read files
packfile="post/packing_fraction.txt"
cundallfile="post/cundall.txt"
zfile="post/z.txt"
#first load packing fraction and cundall to ensure the system is in equilibrium
steps,packing_fraction=read_packing_fraction(packfile) 
steps,c_filtered,c_full=read_cundall(cundallfile)
steps,z=read_mean_coordination_number(zfile)
plt.figure(figsize=(9,3))

plt.subplot(131)
plt.plot(steps, c_full,'r--')
plt.plot(steps,c_filtered,'r-')
plt.title("Cundall full and filtered")
plt.xlabel("t")
plt.ylabel("C")
#plt.legend()
plt.subplot(132)
plt.plot(steps, packing_fraction,'b-')
plt.title("Packing fraction")
plt.xlabel("t")
plt.ylabel("Packing Fraction")
#plt.legend()
plt.subplot(133)
plt.plot(steps,z,'g-')
plt.title("Mean coordination number")
plt.xlabel("t")
plt.ylabel("z")
#plt.legend()
#plt.suptitle('Equilibrium variables')
plt.savefig("plots/equilibrium_parameters.png")
#plt.show()

#Once you know since what timestep the system is balanced, load pressure files for histograms
eqstep=260000 #Timestep for equilibrium of both forces and volumes

hist_p_filt_file="post/histo_nc_press_filtered_"+str(eqstep)+".txt"
hist_p_full_file="post/histo_nc_press_total_"+str(eqstep)+".txt"
p_filt_file="post/pressure_per_particle_filtered_"+str(eqstep)+".txt"
p_full_file="post/pressure_per_particle_total_"+str(eqstep)+".txt"

pBins_filt, N_filt = read_histogram(hist_p_filt_file)
pBins_full, N_full = read_histogram(hist_p_full_file)
N_filt_f=N_filt.copy()
N_filt_f[0]=0
dN_filt=N_filt_f/N_filt.sum()
N_full_f=N_full.copy()
N_full_f[0]=0
dN_full=N_full_f/N_full_f.sum()

p_full=read_pressure(p_full_file)
p_filt=read_pressure(p_filt_file)
#print(p_full)

#Filter out the first data
plt.figure()
plt.plot(pBins_filt,dN_filt,'r-',label='PDF filtered measured')
plt.plot(pBins_full,dN_full,'g-',label='PDF full measured')
plt.xlabel('P')
plt.ylabel('N/N_{total}')

#Now that we got the data we can find the fitting parameters using scipy's curve fit
popt,pcov=curve_fit(k_gamma,pBins_filt,dN_filt)
#print(popt)
plt.plot(pBins_filt,k_gamma(pBins_filt,*popt),'g--',label='filt fit: k=%5.3f,theta=%5.3f'%tuple(popt))
popt_full,pcov_full=curve_fit(k_gamma,pBins_full,dN_full)
#print(popt)
plt.plot(pBins_full,k_gamma(pBins_full,*popt_full),'r--',label='full fit: k=%5.3f,theta=%5.3f'%tuple(popt_full))
plt.legend()
#plt.show()
plt.savefig("plots/PDF_fitting.png")

#Plot histograms using just the pressure per particle files
plt.figure(figsize=(12,6))

plt.subplot(121)
n,bins, patches = plt.hist(x=p_full, bins='auto',color='red',alpha=0.7,rwidth=0.85)
print(bins[1])

plt.grid(axis='y',alpha=0.75)
plt.ylabel("N")
plt.xlabel("P")
plt.title("PDF for full Pressure")

plt.subplot(122)
n2,bins2, patches2 = plt.hist(x=p_filt, bins='auto',color='red',alpha=0.7,rwidth=0.85)

plt.grid(axis='y',alpha=0.75)
plt.ylabel("N")
plt.xlabel("P")
plt.title("PDF for filtered Pressure")

plt.savefig("plots/histograms_pressure.png")
