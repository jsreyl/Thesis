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


equilibrium_params("_8")


#Once you know since what timestep the system is balanced, load pressure files for histograms
eqstep=665000 #Timestep for equilibrium of both forces and volumes

#--------------------------------------------------------------------
#---------------------Pressure Distribution--------------------------
#--------------------------------------------------------------------

def press_histo_plots(eqstep):
    #read files
    hist_p_filt_file="post/histo_nc_press_filtered_"+str(eqstep)+".txt"
    hist_p_full_file="post/histo_nc_press_total_"+str(eqstep)+".txt"
    p_filt_file="post/pressure_per_particle_filtered_"+str(eqstep)+".txt"
    p_full_file="post/pressure_per_particle_total_"+str(eqstep)+".txt"

    pBins_filt, N_filt = read_histogram(hist_p_filt_file)
    pBins_full, N_full = read_histogram(hist_p_full_file)
    #dN_filt=N_filt/N_filt.sum()
    #dN_full=N_full/N_full.sum()
    #Excluiding lowest values might be necessary for fitting
    N_filt_f=N_filt.copy()
    N_filt_f[0]=0
    dN_filt=N_filt_f/N_filt.sum()
    N_full_f=N_full.copy()
    N_full_f[0]=0
    dN_full=N_full_f/N_full_f.sum()

    p_full=read_array(p_full_file)
    p_filt=read_array(p_filt_file)
    #print(p_full)

    #Filter out the first data
    plt.figure()
    plt.plot(pBins_filt,dN_filt,'r-',label='PDF filtered measured')
    plt.plot(pBins_full,dN_full,'g-',label='PDF full measured')
    print(f"Max dN: {dN_full.max()}")
    plt.xlabel('P')
    plt.ylabel('N/N_{total}')

    #Now that we got the data we can find the fitting parameters using scipy's curve fit
    popt,pcov=curve_fit(exponential,pBins_filt,dN_filt)
    #print(popt)
    plt.plot(pBins_filt,exponential(pBins_filt,*popt),'g--',label='filt fit: k=%5.3f,theta=%5.3f'%tuple(popt))
    popt_full,pcov_full=curve_fit(exponential,pBins_full,dN_full)
    #print(popt)
    plt.plot(pBins_full,exponential(pBins_full,*popt_full),'r--',label='full fit: k=%5.3f,theta=%5.3f'%tuple(popt_full))

    plt.legend()
    #plt.show()
    plt.savefig("plots/PDF_fitting.png")

    #Let's calculate the theta and k parameters theorically using Cardenas' aproach
    #k=<p>^2/std(p)^2, theta=std(p)^2/<p>^2
    plt.figure()
    p_avg_full=p_full.mean()
    p_avg_filt=p_filt.mean()
    p_std_full=np.std(p_full)
    p_std_filt=np.std(p_filt)
    k_full = p_avg_full*p_avg_full/(p_std_full*p_std_full)
    theta_full = p_std_full*p_std_full/(p_avg_full*p_avg_full)
    k_filt = p_avg_filt*p_avg_filt/(p_std_filt*p_std_filt)
    theta_filt = p_std_filt*p_std_filt/(p_avg_filt*p_avg_filt)

    plt.plot(pBins_full,k_gamma(pBins_full,k_full,theta_full),'g-o',label='theo full: k=%5.3f,theta=%5.3f'%(k_full,theta_full))
    plt.plot(pBins_filt,k_gamma(pBins_filt,k_filt,theta_filt),'r-o',label='theo filt: k=%5.3f,theta=%5.3f'%(k_filt,theta_filt))
    print(f"Max PDF: {k_gamma(pBins_full,k_full,theta_full).max()}")

    plt.xlabel("P")
    plt.ylabel("PDF")
    plt.legend()
    #plt.show()
    plt.savefig("plots/PDF_theo.png")

    #Rescale and plot the distribution in the same figure
    resForcedParam=k_gamma(pBins_full,k_full,theta_full).max()/dN_full.max()
    print(f"Forced rescaling parameter: {resForcedParam}")
    plt.figure()
    plt.plot(pBins_full,k_gamma(pBins_full,k_full,theta_full),'g-o',label='theo full: k=%5.3f,theta=%5.3f'%(k_full,theta_full))
    plt.plot(pBins_full,resForcedParam*dN_full,'r-',label='PDF full measured')

    plt.xlabel("P")
    plt.ylabel("PDF")
    plt.legend()
    #plt.show()
    plt.savefig("plots/PDF_forced_rescaled.png")

    #Rescale and plot the distribution in the same figure
    #Here we normalize the distribution function using
    #        \sum_{P} N(P)dP=1
    #since all our dP are the same, we can divide our dN by dP for rescaling
    effectivePbin=pBins_full[1]-pBins_full[0]
    resParam=1/effectivePbin
    print(f"Forced rescaling parameter: {resParam}")
    plt.figure()
    plt.plot(pBins_full,k_gamma(pBins_full,k_full,theta_full),'g-o',label='theo full: k=%5.3f,theta=%5.3f'%(k_full,theta_full))
    plt.plot(pBins_full,resParam*dN_full,'r-',label='PDF full measured')

    #Now that we got the data we can find the fitting parameters using scipy's curve fit
    popt,pcov=curve_fit(k_gamma,pBins_full,resParam*dN_full)
    plt.plot(pBins_full,k_gamma(pBins_full,*popt),'b--',label='fit full: k=%5.3f,theta=%5.3f'%tuple(popt))

    plt.xlabel("P")
    plt.ylabel("PDF")
    plt.legend()
    #plt.show()
    plt.savefig("plots/PDF_rescaled.png")


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

#press_histo_plots(eqstep)

#--------------------------------------------------------------------
#---------------------Cell Volume Distribution-----------------------
#--------------------------------------------------------------------

def vol_histo_plots(eqstep):
    #read files
    hist_v_filt_file="post/histo_vorocells_vol_filtered_"+str(eqstep)+".txt"
    hist_v_full_file="post/histo_vorocells_vol_total_"+str(eqstep)+".txt"
    v_filt_file="post/vol_per_vorocell_filtered_"+str(eqstep)+".txt"
    v_full_file="post/vol_per_vorocell_total_"+str(eqstep)+".txt"

    vBins_filt, vc_filt = read_histogram(hist_v_filt_file)
    vBins_full, vc_full = read_histogram(hist_v_full_file)
    dvc_filt=vc_filt/vc_filt.sum()
    dvc_full=vc_full/vc_full.sum()
    #vN_filt_f=N_filt.copy()
    #vN_filt_f[0]=0
    #dN_filt=N_filt_f/N_filt.sum()
    #N_full_f=N_full.copy()
    #N_full_f[0]=0
    #dN_full=N_full_f/N_full_f.sum()

    v_full=read_array(v_full_file)
    v_filt=read_array(v_filt_file)

    #Filter out the first data
    plt.figure()
    plt.plot(vBins_filt,dvc_filt,'r-',label='Voro PDF filtered measured')
    plt.plot(vBins_full,dvc_full,'g-',label='Voro PDF full measured')
    plt.xlabel('Vorovol')
    plt.ylabel('N_cells/N_cells_{total}')
    plt.legend()
    #plt.show()
    plt.savefig("plots/Voro_PDF_fitting.png")

    #Rescale and fit
    #calculate theoretical parameters
    v_min=v_full.min()
    v_avg=v_full.mean()
    v_std=np.std(v_full)
    k_voro= (v_avg-v_min)*(v_avg-v_min)/(v_std*v_std)
    chi_voro=(v_std*v_std)/(v_avg-v_min)
    print("Theoretical voronoi parameters:")
    print(f"k: {k_voro}, chi: {chi_voro}")
    plt.figure()
    plt.plot(vBins_full-v_min,k_gamma(vBins_full-v_min,k_voro,chi_voro),'g-o',label='theo full: k=%5.3f,theta=%5.3f'%(k_voro,chi_voro))
    #Rescale so the distrbution is normalized
    effectiveVbin=vBins_full[1]-vBins_full[0]
    resParamVoro=1/effectiveVbin
    print(f"Rescaling factor for Voronoi: {resParamVoro}")
    plt.plot(vBins_full-v_min,resParamVoro*dvc_full,'r-',label='PDF full voro')
    #and fit
    popt2,pcov2=curve_fit(k_gamma,vBins_full-v_min,resParamVoro*dvc_full)
    plt.plot(vBins_full-v_min,k_gamma(vBins_full-v_min,*popt2),'b--',label='full fit: k=%5.3f,theta=%5.3f'%tuple(popt2))

    plt.xlabel("V-Vmin")
    plt.ylabel("PDF Vorocell")
    plt.legend()
    plt.savefig("plots/Voro_PDF_rescaled.png")
    
    """
    #Now that we got the data we can find the fitting parameters using scipy's curve fit
    popt2,pcov2=curve_fit(k_gamma,vBins_filt,dvc_filt)
    #print(popt)
    plt.plot(vBins_filt,k_gamma(vBins_filt,*popt2),'g--',label='filt fit: k=%5.3f,theta=%5.3f'%tuple(popt2))
popt_full2,pcov_full2=curve_fit(k_gamma,vBins_full,dvc_full)
#print(popt)
    plt.plot(vBins_full,k_gamma(vBins_full,*popt_full2),'r--',label='full fit: k=%5.3f,theta=%5.3f'%tuple(popt_full2))
    """
    
    
    #Plot histograms using just the pressure per particle files
    plt.figure(figsize=(12,6))
    
    plt.subplot(121)
    n_vc,bins_vc, patches_vc = plt.hist(x=v_full, bins='auto',color='red',alpha=0.7,rwidth=0.85)
    print("n_vc: ")
    print(n_vc)
    print("bins_vc: ")
    print(bins_vc)
    print("patches_vc: ")
    print(patches_vc)
    
    plt.grid(axis='y',alpha=0.75)
    plt.ylabel("Cells")
    plt.xlabel("Vorovol")
    plt.title("PDF for full Vorocell")
    
    plt.subplot(122)
    n_vc2,bins_vc2, patches_vc2 = plt.hist(x=v_filt, bins='auto',color='red',alpha=0.7,rwidth=0.85)
    
    plt.grid(axis='y',alpha=0.75)
    plt.ylabel("Cells")
    plt.xlabel("Vorovol")
    plt.title("PDF for filtered Vorovol")

    plt.savefig("plots/histograms_vorovol.png")
    
    #Rescaling with better precision using plt.hist
    #let's use n_vc and bins_vc
    #n_vc is the number of cells with a given Voronoi volume bin
    #bins are the values of volume at the edge of each bin
    
    #first define our normalizing factor
    #sum_V n(V)dV=1
    dVbins=(bins_vc[1]-bins_vc[0])
    resParamVol=1/(dVbins*n_vc.sum())
    print(f"Rescaling factor for hist voronoi: {resParamVol}")
    #Now create a set of points (V,n_vc), the first one we approximate with the average value within a bin
    V_vals=np.zeros(len(n_vc))
    for i in np.arange(len(n_vc)):
        V_vals[i]=0.5*(bins_vc[i+1]+bins_vc[i])
        
    #Now let's get to the nitty-gritty with graphs
    plt.figure()
    v_min=bins_vc.min()
    #Measured and rescaled graph
    print(f"V shape: {V_vals.shape}")
    print(f"n shape: {n_vc.shape}")
    
    plt.plot(V_vals-v_min,resParamVol*n_vc,'r-',label='PDF full hist voro')
    #Theoretical graph
    k_voro= (v_avg-v_min)*(v_avg-v_min)/(v_std*v_std)
    chi_voro=(v_std*v_std)/(v_avg-v_min)
    print("Theoretical voronoi parameters [histogram data]:")
    print(f"k: {k_voro}, chi: {chi_voro}")
    plt.plot(V_vals-v_min,k_gamma(V_vals-v_min,k_voro,chi_voro),'g-o',label='theo full: k=%5.3f,theta=%5.3f'%(k_voro,chi_voro))
    #Fitting
    popt_h,pcov_h=curve_fit(k_gamma,V_vals-v_min,resParamVol*n_vc)
    plt.plot(V_vals-v_min,k_gamma(V_vals-v_min,*popt_h),'b--',label='full fit: k=%5.3f,theta=%5.3f'%tuple(popt_h))
    
    plt.xlabel("V-Vmin")
    plt.ylabel("PDF Vorocell")
    plt.legend()
    plt.savefig("plots/Voro_PDF_rescaled_histo.png")
        
#vol_histo_plots(eqstep)
