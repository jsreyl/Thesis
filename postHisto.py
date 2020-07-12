import sys
import numpy as np
import glob
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
    #Reads files with one line and store all values in an array
    auxlog=INFOPRINT(f"Reading array from file {fname}")
    array = np.loadtxt(fname, unpack=True)
    return array

def read_pressure(fname):
    auxlog=INFOPRINT(f"Reading pressure and coordination number from file {fname}")
    pressure,contacts = np.loadtxt(fname, unpack=True)
    return pressure, contacts

def read_voronoi(fname):
    auxlog=INFOPRINT(f"Reading voronoi info from file {fname}")
    ii,vertices,edges,faces,area,vorovol = np.loadtxt(fname, unpack=True)
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

################################################
############## PLOTTING FUNCTIONS ##############
################################################

def gamma_var_param_histo_plot(var,param,vname,pname,nbins,low_end,high_end,fname,distance,xmin=0.,xmax=0.,autonorm=False):
    """
    This function plots out histograms and fitting graphs for a variable var classified by a parameter param (whose names are vname,pname). It is assumed these are fitted by k-gamma functions.
    For example plot out histograms for the Pressure classifying by number of contacts z, or volume classified by number of faces
    INPUT: var,param,vname,pname,nbins,low_end-high_end,function,fname, distance,xmin-xmax,autonotm -> arrays for variable and parameter (must be the same length), names for variable and paremeter, number of bins for histograms, restrictions on the parameter, name for output files, filtering distance, whether to use plt's automatic normalization.
    OUTPUT: plots for the variable classified by the value of the parameter.
    """
    #First find out how many different values for the parameter we have
    mask_low=param>low_end;mask_high=param<high_end;mask=mask_low & mask_high
    var=var[~np.isinf(var)]
    param_vals=np.unique(param[mask])
    len_param=len(param_vals)
    #create arrays to store data
    n=np.zeros((len_param,nbins))
    bins=np.zeros((len_param,nbins+1))

    #plot histogram for each parameter
    plt.figure()
    if xmin != xmax: #plot using the same scale on x
        plt.xlim(xmin,xmax)
    for i,p in zip(np.arange(len_param),param_vals):
        if xmin != xmax:
            n[i],bins[i],patches=plt.hist(x=var[param==p],bins=nbins,range=[xmin,xmax],density=autonorm,alpha=0.5,rwidth=0.85,label=str(pname)+"="+str(p))
        else:
            n[i],bins[i],patches=plt.hist(x=var[param==p],bins=nbins,density=autonorm,alpha=0.5,rwidth=0.85,label=str(pname)+"="+str(p))
    plt.grid(axis='y',alpha=0.75)
    plt.ylabel('N')
    plt.xlabel(str(vname))
    plt.legend()
    plt.title(str(vname)+" for different "+str(pname))
    plt.savefig(f"plots/histogram_{vname}_{pname}_{fname}.png")
    plt.close()

    #Now rescale and fit to a function
    #rescaling coefficients for each distribution (different values of param)
    if not autonorm: #We must normalize manually, so calculate rescaling factors
        resFactor=np.zeros(len_param)
        for i in np.arange(len_param):
            dBin=bins[i][1]-bins[i][0]
            resFactor[i]=1/(dBin*n[i].sum())
        print(f"Rescaling coefficients for {vname}: {resFactor}")
    else: #It's already normalized
        resFactor=np.ones(len_param)
    
    #Now create a set of points (var,n) for each distribution, aproximating var to the average value in a bin
    var_vals=np.zeros((len_param,nbins))
    for i in np.arange(len_param):
        for j in np.arange(nbins):
            var_vals[i][j]=0.5*(bins[i][j+1]+bins[i][j])
    #For a gamma distribution we can calculate the theoretical k and theta
    #Note these do not depend on whether or not the histograms are normalized since they are calclated from the input arrays.
    k=np.zeros(len_param)
    kprint=np.zeros(high_end-low_end-1)
    theta=np.zeros(len_param)
    thetaprint=np.zeros(high_end-low_end-1)
    for i,p in zip(np.arange(len_param),param_vals):
        var_avg=var[param==p].mean()
        var_std=np.std(var[param==p])
        k[i]=var_avg*var_avg/(var_std*var_std)
        theta[i]=(var_std*var_std)/var_avg
    print(f"Theroretical values for a gamma distribution on variable {vname}")
    print(f"k={k}, theta={theta}")
    #Now store these values and set all others (ones not present in the distribution) to zero
    j=0
    for i in np.arange(low_end+1,high_end):
        if i in param_vals:
            kprint[i-low_end-1]=k[j]
            thetaprint[i-low_end-1]=theta[j]
            j+=1

    #Save to a file
    kafile=f"post/{vname}_k_{distance}d.txt"
    with open(kafile,'w') as kfile:
        auxlog=INFOPRINT(f"Saving k and theta values to file {kafile}.")
       #We need to write at least the distance
        args="{}\n"
        #and then a space for each element in the k array and a space for each element in theta
        for i in np.arange(len(kprint)):
            args="{} {} "+args
        kfile.write(args.format(distance, *kprint, *thetaprint))

    #And finally plot everything
    plt.figure()
    cmap=plt.get_cmap('jet_r')#Color map
    for i,p in zip(np.arange(len_param),param_vals):
        color=cmap(float(i)/len_param)
        #plt.figure()
        print(f"# Attempting plot for {pname}={p}")
        #Measured distribution
        plt.plot(var_vals[i],resFactor[i]*n[i],'o',c=color)
        #plt.plot(var_vals[i],resFactor[i]*n[i],'o',label="PDF for %s=%d"%(pname,p)) #labels can make plotting everything illegible
       #Theoretical gamma distribution
        plt.plot(var_vals[i],k_gamma(var_vals[i],k[i],theta[i]),'-',c=color,label='Theo %s= %d: k=%5.3e,theta=%5.3e'%(pname,p,k[i],theta[i]))
        #Fitting
        #it may be possible that we don't find fitting parameters, so just avoid it and keep plotting
        try:
            popts,pcov=curve_fit(k_gamma,var_vals[i],resFactor[i]*n[i])
            plt.plot(var_vals[i],k_gamma(var_vals[i],*popts),'--',c=color)
            #plt.plot(var_vals[i],k_gamma(var_vals[i],*popts),'--',label='Fit %s= %d: k=%5.3e,theta=%5.3e'%(pname,p,*popts)) #Again only use labels if you're certain it won't hide the plot
        except Exception as e:
            print(e)
            pass
    #if you want a plot per parameter variation, just ident the following lines so they are inside the previous for, also change tha plot name on savefig to plot on different files
    plt.xlabel(str(vname))
    plt.ylabel("PDF "+str(vname))
    plt.legend()
    #plt.savefig(f"plots/PDF_{vname}_{pname}_{p}_{fname}.png")
    plt.savefig(f"plots/PDF_{vname}_{pname}_{fname}.png")
    plt.close()

def write_kfile(outfilename,fnames_pattern):
    """
    This function writes a file outfilename merging the contents of a number of files determined by a pattern
    INPUT: outfilename, fnames_pattern -> name of the output file, pattern to collect an open different files
    OUTPUT: file with the appended contents of all indicated files
    """
    auxlog=INFOPRINT(f"Writing merged file {outfilename}")
    #open all filenames
    fnames=glob.glob(fnames_pattern)
    #sort them so we write in the intended order
    fnames.sort(key=lambda x: int(x.split('_')[-1].split('d')[0]))

    with open(outfilename,'w') as outfile:
        #Iterate through the files
        for name in fnames:
            with open(name) as infile:
                #read the data from each file and write into out file
                outfile.write(infile.read())
                #Add \n to start writing next file from a new line
            outfile.write("\n")

def filtered_params_plots(distance,particlenumber,ks,knames,fname):
    """
    This function plots a number of parameters against a filtering distance
    INPUT: distance, particle number, *Parameters, *ParamNames, fname -> distance array, particle number consistent with the distance filtered, array containing the parameters to be plotted (an array of arrays), names for each of those parametrs and output plot name
    OUTPUT: plots for parameter vs distance and particle number vs distance
    """
    auxlog=INFOPRINT(f'Plotting filtering data')
    plt.figure()
    for k,i in zip(ks,np.arange(len(ks))):
        print(f"Attempting plot for parameter: {knames[i]}")
#    plt.figure(figsize=(9,3*int(len(ks)/3+1)))
#    plt.subplot(131)
        plt.plot(distance,k,'-o',label=f"{knames[i]}")
    plt.xlabel("Separation distance/diameter")
    plt.ylabel("Parameters")
    plt.legend(loc='upper left')
    plt.savefig(fname)
    plt.close()
    plt.figure()
    plt.plot(distance,particlenumber,'-o')
    plt.title("Particle number vs. Filtering distance")
    plt.xlabel("Separation distance/diameter")
    plt.ylabel("Particle number")
    plt.savefig("plots/particlenumber_distance.png")
    plt.close()
    

def k_vs_param_plot(fname,param,pname):
    """
    This function plots a graph relating k with a parameter, for example k vs z or k vs s.
    INPUT: fname,param,pname->name of the file where k values are stored, array containing the parameter values in the same order as they were printed in fname
    OUTPUT: k vs param plot, k and theta arrays
    """
    auxlog=INFOPRINT(f"Plotting k vs {pname}")
    #fname="post/P_k_3d.txt"
    ks=read_array(fname)
    #First value is the filtering distance used, the rest are the k and theta values
    print(f"Filtering distance used: {ks[0]}")
    #remove the distance and keep ks and thetas
    ks=ks[1:]
    #since there's one theta per k, the array should have even length
    print("Checking k array...")
    if len(ks)%2==0:
        print("Seems alright, separating ks from thetas.")
    else:
        print(ERROR+"Array should be of even length (one k per one theta), won\'t process.")
        sys.exit(1)
    k=ks[:int(len(ks)/2)]
    theta=ks[int(len(ks)/2):]
    plt.figure()
    if len(param)!=len(k):
        print(WARNING+"Not enough parameter values, resorting to np.arange.")
        param=np.arange(len(k))
    plt.plot(param,k,'-o',label='Theoretical')
    popts,pcov=curve_fit(linear,param,k)
    plt.plot(param,linear(param,*popts),'--',label="Fit k=a*%s+b, a=%5.2e,b=%5.2e"%(pname,*popts))
    print("Fitted k vs %s to a linear function with a=%5.8f, b=%5.8f"%(pname,*popts))
    plt.xlabel(f"{pname}")
    plt.ylabel("k")
    plt.legend()
    plt.savefig(f"plots/k_{pname}.png")
    plt.close()
    return k, theta

def palpha_z_plot(p,pname,z,zname,zvals,theta,tname):
    """
    This function plots an averaged quantity times its conjugated variable against a parameter.
    INPUT: p,z,zvals,theta->p and z are the variable and parameters, these should be arrays of the sma length. zvals is the values we want to evaluate, theta is an array of the same size as zvals containing the conjugate variable to p
    """
    auxlog=INFOPRINT(f"Plotting <{pname}>*{tname} vs {zname}")
    pavg=np.zeros(len(zvals))
    for i,zv in enumerate(zvals):
        pavg[i]=p[z==zv].mean()
    print(f"Average {pname} for each {zname}: {pavg}")
    plt.figure()
    plt.plot(zvals,pavg/theta,'-o',label="Theroric")
    popts,pcov=curve_fit(linear,zvals,pavg/theta)
    plt.plot(zvals,linear(zvals,*popts),'--',label="Fit <%s>*%s=a*%s+b, a=%5.2e,b=%5.2e"%(pname,tname,zname,*popts))
    print("Fitted <%s>*%s vs %s to a linear function with a=%5.8f, b=%5.8f"%(pname,tname,zname,*popts))
    plt.xlabel(f"{zname}")
    plt.ylabel(f"<{pname}>*{tname}")
    plt.legend()
    plt.savefig(f"plots/{pname}avg_{zname}.png")
    plt.close()


################################################
############# FITTING FUNCTIONS ################
################################################

def k_gamma(x,k,theta):
    return gamma.pdf(x,a=k,scale=theta)

def exponential(x,a,b):
    return a*np.power(x,b)

def linear(x,a,b):
    return a*x+b

################################################
############ MAIN IMPLEMENTATION ###############
################################################

#first load packing fraction and cundall to ensure the system is in equilibrium

#--------------------------------------------------------------------
#--------------------Equilibrium Parameters--------------------------
#--------------------------------------------------------------------

def equilibrium_params(steps,c_full,c_filtered,packing_fraction,z,nname):
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
    plt.close()

#Find equilibrium time
def find_equilibrium_time(steps, c_full,c_filtered,packing_fraction,z,start_time):
    #variables for storing equilibrium times for each of the parameters
    mask=steps>start_time
    eqsteps=np.zeros(4)
    ERR=5e-8
    for i in np.arange(1,len(steps[mask])):
        if (((c_full[mask][i]-c_full[mask][i-1])<= ERR) and (eqsteps[0]==0.)):
            eqsteps[0] =steps[mask][i]
        if (((c_filtered[mask][i]-c_filtered[mask][i-1])<= ERR) and (eqsteps[1]==0.)):
            eqsteps[1] =steps[mask][i]
        if (((packing_fraction[mask][i]-packing_fraction[mask][i-1])<= ERR) and (eqsteps[2]==0.)):
            eqsteps[2] =steps[mask][i]
        if (((z[mask][i]-z[mask][i-1])<= ERR) and (eqsteps[3]==0.)):
            eqsteps[3] =steps[mask][i]
        if (np.all(eqsteps!=0.)):
            break;
    print(f"Equilibrium step for cundall full: {eqsteps[0]}")
    print(f"Equilibrium step for cundall filt: {eqsteps[1]}")
    print(f"Equilibrium step for packingfraction: {eqsteps[2]}")
    print(f"Equilibrium step for coordination numeber: {eqsteps[3]}")
    eqstep=eqsteps.max()
    return int(eqstep)


#read files
nname=""
packfile="post/packing_fraction"+str(nname)+".txt"
cundallfile="post/cundall"+str(nname)+".txt"
zfile="post/z"+str(nname)+".txt"

steps,packing_fraction=read_packing_fraction(packfile)
steps,c_filtered,c_full=read_cundall(cundallfile)
steps,z=read_mean_coordination_number(zfile)

equilibrium_params(steps,c_full,c_filtered,packing_fraction,z,nname)

eqstep_f=find_equilibrium_time(steps,c_full,c_filtered,packing_fraction,z,100000)
print(f"Equilibrium time: {eqstep_f}")

#Once you know since what timestep the system is balanced, load pressure files for histograms
#eqstep=3*eqstep_f #Timestep for equilibrium of both forces and volumes
eqstep=670000


#--------------------------------------------------------------------
#---------------------Pressure Distribution--------------------------
#--------------------------------------------------------------------

#read files
ndist=20
#----- Pressure histograms filtering with distance-------------------
for dist in np.arange(ndist):
    p_file=f"post/pressure_per_particle_filtered{dist}d_{eqstep}.txt"
    p,z=read_pressure(p_file)
    print(f"Number of particles filtering for distance {dist}: {len(p[p!=0.])}")
    gamma_var_param_histo_plot(p[p!=0.],z[p!=0.],"P","z",60,3,9,f"filt{dist}d",dist,xmin=0.,xmax=0.08)


#Now write all k values on the same file
kfilename="post/P_k.txt"
kfile_pattern="post/P_k_*d.txt"
write_kfile(kfilename,kfile_pattern)

#read distance vs parameter files
print(INFO+f"Reading packing fraction file")
distance,particlenumber,packing_fraction,packing_vorovol=np.loadtxt("post/packing_fraction_distance.txt",unpack=True)
print(INFO+f"Reading q/p file")
distance,qp=np.loadtxt("post/qp_distance.txt",unpack=True)
print(INFO+f"Reading pressure k file")
distance,k4,k5,k6,k7,k8,t4,t5,t6,t7,t8=np.loadtxt("post/P_k.txt",unpack=True)
print("Done reading!")
ks=[packing_fraction[distance<19],packing_vorovol[distance<19]]
knames=["packing_fraction","packing_fraction_voro"]
filtered_params_plots(distance[distance<19],particlenumber[distance<19],ks,knames,"plots/Param_packing_fraction_distance_18omenos.png")
ks=[qp[distance<19]]
knames=["q/p"]
filtered_params_plots(distance[distance<19],particlenumber[distance<19],ks,knames,"plots/Param_qp_distance_18omenos.png")
ks=[k4[distance<19],k5[distance<19],k6[distance<19],k7[distance<19],k8[distance<19]]
knames=["k(z=4)","k(z=5)","k(z=6)","k(z=7)","k(z=8)"]
filtered_params_plots(distance[distance<19],particlenumber[distance<19],ks,knames,"plots/Param_Pressure-k_distance_18omenos.png")

#--------------k vs z plot---------------------------------
kname="post/P_k_3d.txt"
zs=np.array([4,5,6,7,8])
k_p,ang=k_vs_param_plot(kname,zs,"z")

#------------<P>*alpha=(a*z+b)-----------------------------
p_file=f"post/pressure_per_particle_filtered3d_{eqstep}.txt"
p,z=read_pressure(p_file)
palpha_z_plot(p[p!=0],"P",z[p!=0],"z",zs,ang,"alpha")


#---------------------------------------------------------------------
#---------------------Cell Volume Distribution------------------------
#---------------------------------------------------------------------

#Theoretical value used by Oquendo on his 2016 paper
#V_min/v_grain \approx 1.3250
#See page 183 Aste T. and Weaire D., "The Pursuit of Perfect Packing", 2008 for mor info

rad_file="post/rad_"+str(eqstep)+".txt"
rad = read_array(rad_file)
v_grain=4.*np.pi*(rad.mean()**3)/3.
v_min=1.3250*v_grain

#------Volume histograms filtering distance-----------------
for dist in np.arange(ndist):
    v_file=f"post/voro_ixyz_filtered{dist}d_{eqstep}.txt.vol"
    v,w,g,s,area=read_voronoi(v_file)
    print(f"Number of particles filtering for distance {dist}: {len(v)}")
    gamma_var_param_histo_plot(v-v_min,s,"V-Vmin","s",60,11,20,f"filt{dist}d",dist,xmin=0.,xmax=1.2e-7)

#Now write all k values on the same file
kfilename="post/V-Vmin_k.txt"
kfile_pattern="post/V-Vmin_k_*d.txt"
write_kfile(kfilename,kfile_pattern)

#read distance vs parameter files
distance,k12,k13,k14,k15,k16,k17,k18,k19,t12,t13,t14,t15,t16,t17,t18,t19=np.loadtxt("post/V-Vmin_k.txt",unpack=True)
ks=[k12[distance<17],k13[distance<17],k14[distance<17],k15[distance<17],k16[distance<17],k17[distance<17],k18[distance<17],k19[distance<17]]
knames=["k(s=12)","k(s=13)","k(s=14)","k(s=15)","k(s=16)","k(s=17)","k(s=18)","k(s=19)"]
filtered_params_plots(distance[distance<17],particlenumber[distance<17],ks,knames,"plots/Param_Volume-k_distance_16omenos.png")

#--------Full distribution for volume--------------
print("### Calculating distibution for all volumes")
v_file="post/voro_ixyz_filtered3d_670000.txt.vol"
v,w,g,s,area=read_voronoi(v_file)
xmin=0.
xmax=1.2e-7
plt.figure()
plt.xlim(xmin,xmax)
n,bins,patches=plt.hist(v-v_min,bins=60,range=[xmin,xmax],alpha=0.5,rwidth=0.85,label="All Vorovol")
plt.ylabel('N')
plt.xlabel("V-Vmin")
plt.legend()
plt.savefig(f"plots/histogram_V-Vmin_all.png")
plt.close()

#Now rescale
dBin=bins[1]-bins[0]
resFactor=1/(dBin*n.sum())
print(f"Rescaling factor: {resFactor}")
vol_vals=np.zeros(60)
for i in np.arange(60):
    vol_vals[i]=0.5*(bins[i+1]+bins[i])
vol_avg=(v-v_min).mean()
vol_std=np.std(v-v_min)
k=vol_avg*vol_avg/(vol_std*vol_std)
theta=(vol_std*vol_std)/vol_avg
print(f"Theoretical values for all volumes")
print(f"k:{k}, theta={theta}")
plt.figure()
plt.xlim(xmin,xmax)
plt.plot(vol_vals,resFactor*n,'o',label="Normalized manually")
plt.plot(vol_vals,k_gamma(vol_vals,k,theta),'-',label="Theo k=%5.3e, theta=%5.3e"%(k,theta))
n,bins,patches=plt.hist(v-v_min,bins=60,range=[xmin,xmax],alpha=0.5,rwidth=0.85,density=True,label="Normalized by plt")
plt.xlabel("V-Vmin")
plt.ylabel("PDF for all vorovol")
plt.legend()
plt.savefig("plots/PDF_V-Vmin_all.png")
plt.close()

#-----k vs s plot-----------------------------------
k2name="post/V-Vmin_k_3d.txt"
ss=np.array([12,13,14,15,16,17,18,19])
k_v,chi=k_vs_param_plot(k2name,ss,"s")

#-----(<V>-Vmin)*1/chi=(a*s+b)----------------------
palpha_z_plot(v-v_min,"V-Vmin",s,"s",ss,chi,"1/chi")
