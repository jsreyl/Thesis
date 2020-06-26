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


################################################
################# READ FILES ###################
################################################

def read_histogram(fname):
    auxlog=INFOPRINT(f"Reading histogram from file {fname}")
    Bins, nums = np.loadtxt(fname, unpack=True)
    return Bins, nums

def read_array(fname):
    auxlog=INFOPRINT(f"Reading array from file {fname}")
    array = np.loadtxt(fname, unpack=True)
    return array

def read_pressure(fname):
    auxlog=INFOPRINT(f"Reading pressure and coordination number from file {fname}")
    pressure,contacts = np.loadtxt(fname, unpack=True)
    return pressure, contacts

def read_voronoi(fname):
    auxlog=INFOPRINT(f"Reading voronoi info from file {fname}")
    ii,x,y,z,vertices,edges,faces,area,vorovol = np.loadtxt(fname, unpack=True)
    return vorovol,vertices,edges,faces,area

def read_cundall(fname):
    auxlog=INFOPRINT(f"Reading cundall data from file {fname}")
    timestep, cundall_filtered, cundall_full = np.loadtxt(fname, unpack=True)
    return timestep, cundall_filtered, cundall_full

def read_mean_coordination_number(fname):
    auxlog=INFOPRINT(f"Reading mean coordination number data from file {fname}")
    timestep, z = np.loadtxt(fname, unpack=True)
    return timestep, z

def read_packing_fraction(fname):
    auxlog=INFOPRINT(f"Reading packing fraction data from file {fname}")
    timestep, packing_fraction = np.loadtxt(fname, unpack=True)
    return timestep, packing_fraction

def gamma_var_param_histo_plot(var,param,vname,pname,nbins,low_end,high_end,fname):
    """
    This function plots out histograms and fitting graphs for a variable var classified by a parameter param (whose names are vname,pname).
    INPUT: var,param,vname,pname,nbins,low_end,high_end,function,fname -> arrays for variable and parameter (must be the same length), names for variable and paremeter, number of bins for histograms, restrictions on the parameter, a function for fitting, name for output files
    OUTPUT: plots for the variable classified by the value of the parameter.
    """
    #First find out how many different values for the parameter we have
    mask_low=param>low_end;mask_high=param<high_end;mask=mask_low & mask_high
    param_vals=np.unique(param[mask])
    len_param=len(param_vals)
    #create arrays to store data
    n=np.zeros((len_param,nbins))
    bins=np.zeros((len_param,nbins+1))

    #plot histogram for each parameter
    plt.figure()
    for i,p in zip(np.arange(len_param),param_vals):
        n[i],bins[i],patches=plt.hist(x=var[param==p],bins=nbins,alpha=0.5,rwidth=0.85,label=str(pname)+"="+str(p))
    plt.grid(axis='y',alpha=0.75)
    plt.ylabel('N')
    plt.xlabel(str(vname))
    plt.legend()
    plt.title(str(vname)+" for different "+str(pname))
    plt.savefig("plots/histogram_"+(vname)+"_"+str(pname)+"_"+str(fname)+".png")

    #Now rescale and fit to a function
    #rescaling coefficients for each distribution (different values of param)
    resParams=np.zeros(len_param)
    for i in np.arange(len_param):
        dBin=bins[i][1]-bins[i][0]
        resParams[i]=1/(dBin*n[i].sum())
    print(f"Rescaling coefficients for {vname}: {resParams}")
    #Now create a set of points (var,n) for each distribution, aproximating var to the average value in a bin
    var_vals=np.zeros((len_param,nbins))
    for i in np.arange(len_param):
        for j in np.arange(nbins):
            var_vals[i][j]=0.5*(bins[i][j+1]+bins[i][j])
    #For a gamma distribution we can calculate the theoretical k and theta
    k=np.zeros(len_param)
    theta=np.zeros(len_param)
    for i,p in zip(np.arange(len_param),param_vals):
        var_avg=var[param==p].mean()
        var_std=np.std(var[param==p])
        k[i]=var_avg*var_avg/(var_std*var_std)
        theta[i]=(var_std*var_std)/var_avg
    print(f"Theroretical values for a gamma distribution on variable {vname}")
    print(f"k={k}, theta={theta}")

    #And plot
    for i,p in zip(np.arange(len_param),param_vals):
        plt.figure()
        #Measured distribution
        plt.plot(var_vals[i],resParams[i]*n[i],'o',label="PDF for %s=%d"%(pname,p))
        #Theoretical gamma distribution
        plt.plot(var_vals[i],k_gamma(var_vals[i],k[i],theta[i]),label='Theo %s= %d: k=%5.3f,theta=%5.3f'%(pname,p,k[i],theta[i]))
        #Fitting
        popts,pcov=curve_fit(k_gamma,var_vals[i],resParams[i]*n[i])
        plt.plot(var_vals[i],k_gamma(var_vals[i],*popts),'--',label='Fit %s= %d: k=%5.3f,theta=%5.3f'%(pname,p,*popts))
        plt.xlabel(str(vname))
        plt.ylabel("PDF "+str(vname))
        plt.legend()
        plt.savefig("plots/PDF_"+str(vname)+"_"+str(pname)+"_"+str(p)+"_"+str(fname)+"_rescaled.png")


################################################
############# FITTING FUNCTIONS ################
################################################

def k_gamma(x,k,theta):
    return gamma.pdf(x,a=k,scale=theta)

def exponential(x,a,b):
    return a*np.power(x,b)

################################################
############ MAIN IMPLEMENTATION ###############
################################################

#first load packing fraction and cundall to ensure the system is in equilibrium

#--------------------------------------------------------------------
#--------------------Equilibrium Parameters--------------------------
#--------------------------------------------------------------------

def equilibrium_params(nname):
    #read files
    packfile="post/packing_fraction"+str(nname)+".txt"
    cundallfile="post/cundall"+str(nname)+".txt"
    zfile="post/z"+str(nname)+".txt"

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
    plt.savefig("plots/equilibrium_parameters"+str(nname)+".png")
    #plt.show()


#equilibrium_params("_8")


#Once you know since what timestep the system is balanced, load pressure files for histograms
eqstep=665000 #Timestep for equilibrium of both forces and volumes

#--------------------------------------------------------------------
#---------------------Pressure Distribution--------------------------
#--------------------------------------------------------------------

#read files
p_filt_file="post/pressure_per_particle_filtered_"+str(eqstep)+".txt"
#p_full_file="post/pressure_per_particle_total_"+str(eqstep)+".txt"
#p_full,z_full=read_pressure(p_full_file)
p_filt,z_filt=read_pressure(p_filt_file)
gamma_var_param_histo_plot(p_filt,z_filt,"P","z",60,2,8,"filt")

#--------------------------------------------------------------------
#---------------------Cell Volume Distribution-----------------------
#--------------------------------------------------------------------

#read files
#v_filt_file="post/vol_per_vorocell_filtered_"+str(eqstep)+".txt"
v_filt_file="post/voro_ixyzr_"+str(eqstep)+".txt.vol"
#v_full_file="post/vol_per_vorocell_total_"+str(eqstep)+".txt"
#v_full,w_full,g_full,s_full,area_full=read_voronoi(v_full_file)
v_filt,w_filt,g_filt,s_filt,area_filt=read_voronoi(v_filt_file)

#Use theoretical Vmin instead, in the meantime use the minumun value for each distribution
s_vals=np.unique(s_filt)
v_min=np.zeros(len(s_vals))
for i,s in zip(np.arange(len(s_vals)),s_vals):
    v_min[i]=v_filt[s_filt==s].min()

for i,s in zip(np.arange(len(s_vals)),s_vals):
    for j in np.arange(len(v_filt)):
        if s_filt[j]==s:
            v_filt[j]=v_filt[j]-v_min[i]

gamma_var_param_histo_plot(v_filt,s_filt,"V-Vmin","faces",60,0,100,"filt")
