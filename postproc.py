import os
import sys
import numpy as np
import glob
import matplotlib.pyplot as plt
from colorama import Fore, Back, Style
WARNING=Fore.YELLOW + '# WARNING: ' + Style.RESET_ALL
INFO=Fore.GREEN + '# INFO: ' + Style.RESET_ALL
ERROR=Fore.RED + '# ERROR: ' + Style.RESET_ALL


class INFOPRINT:
    def __init__(self, message):
        self.message_ = message
        print(INFO + "START -> " + self.message_)
    def __del__(self):
        print(INFO + "DONE  -> " + self.message_ + "\n")


#--------------Functions to read dump files---------------------
#Due to the nature of the dump files, the info is divided in three parts
# One load for box limit info
# One load for total force info (this is calculated for all particles so it can be dumped in a single file)
# One load for contact force info (this is caclulated in a pair wise way so it must be dumped in another file, because the array size is different to that of total force for all particles)
def read_limit_info(fname):
    """
    This function reads a dump file and returns the region limits of the system
    INPUT: fname -> filename of a LIGGGHTS dump file, can be either all or local, the relevan info is in the first rows
    OUTPUT: two arrays with the box limits coord_min and coord_max
    """
    #box limits(rows 6 to 8)
    coord_min, coord_max =np.loadtxt(fname,skiprows=5,max_rows=3, unpack=True)
    return coord_min, coord_max

def read_particle_data(fname):
    """
    This function reads a dump file and returns physical info on all particles of the packing
    INPUT: fname -> filename of the dumped file by LIGGGHTS, these are named packing_gz/packing_[timestep].gz
    OUTPUT: arrays with particle info ID,TYPE,mass,x,y,z,vx,vy,vz,fx,fy,fz,r,omegax,omegay,omegaz,tqx,tqy,tqz
    """
    auxlog=INFOPRINT("reading particle data")
    print("...Reading particle dump file: "+fname+"...")
    #from 10 onwards are the data
    ID,TYPE,mass,x,y,z,vx,vy,vz,fx,fy,fz,r,omegax,omegay,omegaz,tqx,tqy,tqz = np.loadtxt(fname, skiprows=9, unpack=True)
    print(f"Number of particles read: {ID.size}")
    ID = ID.astype(np.int32)
    TYPE = TYPE.astype(np.int32)
    ID=ID-1
    print("ID indexes for particles shifted by -1 so they start at 0")
    #sorting IDs
    #use argsort, which returns the indexes that would sort the ID array and use these argsorted list to sort the other ones without mixing the physical data corresponding to each indexed particle
    IDarray = np.argsort(ID)
    return ID[IDarray], TYPE[IDarray],mass[IDarray],x[IDarray],y[IDarray],z[IDarray],vx[IDarray],vy[IDarray],vz[IDarray],fx[IDarray],fy[IDarray],fz[IDarray],r[IDarray],omegax[IDarray],omegay[IDarray],omegaz[IDarray],tqx[IDarray],tqy[IDarray],tqz[IDarray]


def read_contact_data(fname):
    """
    This function reads a dump local file and returns info on contact forces in pairs of particles
    INPUT: fname -> filename of local contact data dumped by LIGGGHTS, named packing_gz/packing_local[timestep].gz
    OUTPUT: arrays with info positions of particles, ids, contact_force, normal_contact_force, tangential_contact_force, torque, contactArea, interpenetration_distance, contact_point
    """
    auxlog=INFOPRINT("reading contacts")
    print("...Reading contact dump file: "+fname+"...")
    x1,y1,z1,x2,y2,z2,ID1,ID2,IDperiod,fcx,fcy,fcz,fnx,fny,fnz,ftx,fty,ftz,tx,ty,tz,cArea,delta,cx,cy,cz=np.loadtxt(fname, skiprows=9, unpack=True)
    print(f"Number of particles read: {x1.size}")
    ID1 = ID1.astype(np.int32)
    ID2 = ID2.astype(np.int32)
    ID1=ID1-1
    ID2=ID2-1
    print("ID indexes for particles shifted by -1 so they start at 0")
    return x1,y1,z1,x2,y2,z2,ID1,ID2,IDperiod,fcx,fcy,fcz,fnx,fny,fnz,ftx,fty,ftz,tx,ty,tz,cArea,delta,cx,cy,cz


#-----------Postprocessing functions-----------------------------

def count_particle_contacts(ID,ID1,ID2):
    """
    This function returns the number of contacts per particle
    INPUT: ID, ID1, ID2 -> arrays of ID for all particles, and ID1, ID2 ids for pairs of particles in contact
    OUTPUT: full_contacts,z_avg -> arrray containing the contacts per particle, ordered by index, the average number of contacts per particle
    """
    auxlog=INFOPRINT("Calculating contacts per particle")
    #first count the number of contacts for the particles with contacts, that is the pairs with ID1 and ID2.
    #However they may be repeated if the particle's got more than one contact so we must count them but once and keep in mind how many times have they appeared (i.e. the number of contacts)
    contact_ID, counts = np.unique(np.concatenate((ID1,ID2)),return_counts=True)
    #contact_ID refers to the ID of a particle with contacts and counts refers to the number of contacts for that ID
    #There may also be particles without contacts, let's add them
    full_contacts=np.zeros_like(ID)
    #here all contacts are set to zero, we must fill the respective count values for the particles in counts
    for i, IDval in enumerate(contact_ID):
        IDmatch = np.where(ID == IDval)#that is, the place where the ID matches an ID of the particle with contacts
        if IDmatch[0].size != 1:#If there's more than one place or none there must be something wrong
            print(ERROR+" Multiple indexes or none found.")
            print("Contact index "+i)
            print("Places found: "+np.count_nonzero(IDmatch))
            print("Index match vector: "+IDmatch)
            sys.exit(1)
        full_contacts[IDmatch]=counts[i]
    z=int(full_contacts.sum()/2)
    z_avg=z/full_contacts.size
    print(INFO+f"full_contacts.size (should be equal to particle number): {full_contacts.size}")
    print(INFO+f"sum(full_contact)/2 (should be equal to number of contacts): {z}")
    print(INFO+f"sum(full_contact)/(2*full_contact.size) (should be equal to average number of contacts): {z_avg}")
    print(INFO+f"a priori average contact number:{ID1.size/ID.size}")
    if z != ID1.size or z != ID2.size:
        print(ERROR+"sum of contact counts does not equal number of contacts")
        print(f"ID1.size={ID1.size}")
        print(f"ID2.size={ID2.size}")
    return full_contacts, z_avg


def mean_coordination_number(mask_pos,mask_pos_contacts):
    """
    This function calculates the mean coordination number (average number of contacts) using masks on particle positions (x,y,z) and contact positions (cx,cy,cz) both of those are dumped by LIGGGHTS and read on the respective functions
    INPUT: masks for particle positions and contact positions
    OUTPUT: mean coordination number
    """
    auxlog=INFOPRINT("Computing mean coordination number")
    #if a particle and a contact are in our region of interest, that is indicated as a 1 on our masks, so simply count how many 1s are there
    NP=np.count_nonzero(mask_pos)
    NC=np.count_nonzero(mask_pos_contacts)
    return 2*NC/NP

#-----------Equilibrium parameters-------------------------------
"""
Force ensemble:
To check for the equilibrium of the force ensemble we use the the Cundall Parameter defined as
C=\sum_p |F_T|/\sum_p |F_c|
where: F_T is the total force
       F_c is the contact force
       and the sum is done over all the particles
Which should go to zero when the system is in mechanical equilibrium
"""

def cundall(fx,fy,fz,cfx,cfy,cfz, mask_pos, mask_pos_contacts):
    """
    This function calculates the cundal parameter for a system
    INPUT: \vec{f},\vec{f_c},masks -> arrays with the total force on each particle and the contact force for each pair of particles, masks to define the region of interest for particles and contacts
    OUTPUT: the cundall parameter for the area of interest and all the packing
    """
    cundall_filtered=np.sum(np.sqrt(np.power(fx[mask_pos],2)+np.power(fy[mask_pos],2)+np.power(fz[mask_pos],2)))/np.sum(np.sqrt(np.power(cfx[mask_pos_contacts],2)+np.power(cfy[mask_pos_contacts],2)+np.power(cfz[mask_pos_contacts],2)))
    cundall_full=np.sum(np.sqrt(np.power(fx,2)+np.power(fy,2)+np.power(fz,2)))/np.sum(np.sqrt(np.power(cfx,2)+np.power(cfy,2)+np.power(cfz,2)))
    return cundall_filtered,cundall_full

"""
Volume ensemble:
To check for the equilibrium of the volume ensemble we use the Packing Fraction:
\phi = particle_volume/total_volume
Which should become a constant close to the Random Close Packing (RCP=0.61)for sphere poured into a bed, when the system is fully compressed. 
"""
def get_box_boundaries(x,y,z,r,screen=np.zeros(3)):
    """
    This function returns the box boundaries
    INPUT: x,y,z,r,screen-> arrays for particle positions, radius and screening variables(these reduce the volume of interest)
    OUTPUT: limits -> an array containing the min and max coordinates for the box boundaries
    """
    limits=np.zeros(6)
    limits[0]= min(x-r)+screen[0]; limits[1]=max(x+r)-screen[0];
    limits[2]= min(y-r)+screen[1]; limits[3]=max(x+r)-screen[1];
    limits[4]= min(x-r)+screen[2]; limits[5]=max(x+r)-screen[2];
    return limits

def packing_fraction(x,y,z,r,limits,mask_pos,screen=np.zeros(3)):
    """
    This function calculates the packing fraction of a rectangular packing
    INPUT: x,y,z,r, limits, mask_pos, screen -> arrays of particle position, box limits, position mask that defines the particles of interest, screen that defines the region of interest (+-x,+-y,+-z to the usual box boundaries for the total volume)
    OUTPUT: packing fraction
    """
    auxlog=INFOPRINT("Calculating packing fraction from box limits")
    #first get the box boundaries and calculate the volume of the box
    limits=get_box_boundaries(x,y,z,r,screen)
    totalVolume=(limits[1]-limits[0])*(limits[3]-limits[2])*(limits[5]-limits[4])
    #now calculate the particle volume on the region of interest defined by the position mask (inside the box, or inside the screen region)
    #we assume al particles are spherical
    particleVolume=4.*np.pi*np.sum(np.power(r[mask_pos],3))/3.
    return particleVolume/totalVolume

def get_mask(x,y,z,radius,limits):
    """
    This function computes a position mask 
    INPUT: x,y,z,r, limits -> arrays of particle position, box limits, position mask that defines the particles of interest, screen that defines the region of interest (+-x,+-y,+-z to the usual box boundaries for the total volume)
    OUTPUT: mask array
    """
    auxlog=INFOPRINT("Computing position masks ")
    #compute mask for particles/contacts inside limits
    masks=[x>=limits[0],x<=limits[1],y>limits[2],y<=limits[3],z>=limits[4],z<=limits[5]]
    #the particle ID is only valid if all region constraints are
    mask=masks[0] & masks[1] & masks[2] & masks[3] & masks[4] & masks[5]
    return mask

#---------------Histograms--------------------------------------------
"""
We need two histograms
1. Contact force per grain (or rather pressure per grain)
2. Volume distributions of Voronoï cells (using voro++)
"""
def pressure_per_particle(ID, ID1, ID2, cfx,cfy,cfz,mask_pos_contacts):
    """
    This function calculates the pressure for each particle in the packing, defined as
    P = \sum_{i=1}^{z} |fn_i|
    where z is the contact number, fn the contact (normal) force between that particle and its contact
    INPUT: IDs, cf-> IDs of all particles and particles in contact pairs, contact force arrays and a mask to select only particles in a position and force range
    OUTPUT: P -> an array containing the pressure for each particle 
    """
    auxlog=INFOPRINT("Calculating pressure for each particle")
    #first calculate the pressure on particles with contacts
    contact_ID = np.unique(np.concatenate((ID1[mask_pos_contacts],ID2[mask_pos_contacts])))
    #array for storing
#    print(f"Contact IDs: {contact_ID}")
    pressure_contacts=np.zeros(contact_ID.size)
    for i, IDval in enumerate(contact_ID):
        IDmatch = np.where((IDval==ID1) | (IDval==ID2)) #each time this particle appears on the contact list
#        print(f"IDmatch: particle {IDval} has the contacts indexed by {IDmatch} connecting {ID1[IDmatch]},{ID2[IDmatch]}")
        fn = np.sqrt(np.power(cfx[IDmatch],2)+np.power(cfy[IDmatch],2)+np.power(cfz[IDmatch],2))
        pressure_contacts[i]=np.sum(fn)
    #now we've got a relation contact_ID<->pressure_contacts
    
    #there may be however particles with no contacts, so let's set them to zero and store all data on a single array
    pressure_full=np.zeros(ID.size)
    for i, IDval in enumerate(contact_ID):
        IDmatch = np.where(IDval == ID)
        if IDmatch[0].size != 1:#If there's more than one place or none there must be something wrong
            print(ERROR+" Multiple indexes or none found.")
            print("Contact index "+i)
            print("Places found: "+np.count_nonzero(IDmatch))
            print("Index match vector: "+IDmatch)
            sys.exit(1)
        pressure_full[IDmatch]=pressure_contacts[i]
    return pressure_full
    
def histogram_pressure_per_particle(press,step,autonbins=True,nbins=20,fname=""):
    """
    This function takes info from a pressure array and outputs a file containing the bins to graph a histogram
    INPUT: press, step, mask_pos, autonbins, nbins -> an array containing the pressure of each particle, time step of the simulation, mask for filtering particles, a bool for letting numpy calculate the optimal bins for the histogram, if falsee specify the number of bins
    OUTPUT: post/histo_nc_press{fname}_{step}.txt: a file containing pressure bins and number of contacts on that bin
    """
    auxlog=INFOPRINT("Calculating histogram pressure per particle")
    if autonbins:
        Nhist,Pbin_edges=np.histogram(press,bins='auto')
    else:
        Nhist,Pbin_edges=np.histogram(press,bins=nbins)

    #If we want an histogram, it's easier to use matplotlib's hist, however numpy can be useful for creating a graph, in which case we can use the half-bin-value to average the pressure per particle.
    Phalf=np.zeros(len(Pbin_edges)-1)
    for i in np.arange(len(Pbin_edges)-1):
        Phalf[i]=0.5*(Pbin_edges[i]+Pbin_edges[i+1])
    Nhist=np.nan_to_num(Nhist)
    np.savetxt(f"post/histo_nc_press{fname}_{step}.txt",np.column_stack((Phalf,Nhist)))

def print_pressure(press,step,fname):
    """
    This function prints a file with pressure per particle info to be used by plt's histogram
    INPUT: press, step, fname -> array containing pressure calcuulated for each particle, timestep of the simulation and fname an additional name indicator for the file
    OUTPUT: post/pressure_per_particle{fname}_{step}.txt : file containing the pressure array
    """
    auxlog=INFOPRINT("Printing pressure to file")
    np.savetxt(f"post/pressure_per_particle{fname}_{step}.txt",np.column_stack((press)))


def voronoi_volume(x,y,z,rad,step, reuse_vorofile=False):
    """
    This function takes the particle postion info and uses voro++ to calculat the voronoi cell volumes as well as creating .pov files to visualize the distribution using povray
    INPUT: x,y,z,rad,step,reuse_vorofile -> arrays containing the position of each partivle, timestep of the simulation and whether to reuse a pre existing voro file
    OUTPUT: vorovol -> array containing voronoi volumes indexed as particles
            voro_ixyzr.txt.vol, voro_ixyzr.txt_p.pov, voro_ixyzr.txt_v.pov: files for processing
    """
    auxlog=INFOPRINT("Computing Voronoi cell volumes using voro++ on command line")
    #To use voro++ we need a file formated as ID x y z rad
    FNAME="post/voro_ixyzr_"+str(step)+".txt"
    #We might use these files later on a cpp code for voro++ to generate .pov images and an animation
    vorofile_exists= os.path.isfile(FNAME+".vol")
    if vorofile_exists and reuse_vorofile:
        print(WARNING + "Reading voro data from existing file: "+FNAME+".vol")
    else:
        print("# Saving coordinates on .vol file")
        #file should be formated as <id> <x> <y> <z>
        #in case of polydisperse distribution also add <r> and add the argument -r to voro++ below
        np.savetxt(FNAME, np.column_stack((np.arange(len(x),dtype=np.int64),x,y,z)),fmt="%i %.18e %.18e %.18e")
        #arguments for voro++ on command line
        #voro++ <args> <xmin> <xmax> <ymin> <ymax> <zmin> <zmax> <filename>
        #for more info http://math.lbl.gov/voro++/doc/cmd.html
        #we're using -v verbose and -o ordering so the output file uses the same ids as the particles
        #other useful args are -r for polydisperse distributions, -p for periodic boundaries and -y for printing .pov files for particles and voro cells
        args = " -v -o -y "+str((x-rad).min())+" "+str((x+rad).max())+" "+str((y-rad).min())+" "+str((y+rad).max())+" "+str((z-rad).min())+" "+str((z+rad).max())+" "+FNAME
        print("# Calling voro++ (this might take a few minutes ...) ")
        print(f"# args for voro++: {args}")
        os.system("voro++ "+args)
    #Now that we have our .vol file load it
    IFNAME=FNAME+".vol"
    print(f"# Loading and filtering voro data from {IFNAME} ...")
    ii,xvoro,yvoro,zvoro,vorovol=np.loadtxt(IFNAME, unpack=True)
    print("# Checking voro data ...")
    if xvoro.size != x.size:
        print(ERROR + "total data read is different from original data")
        print(ERROR + f"x.size:     {x.size}")
        print(ERROR + f"xvoro.size: {xvoro.size}")
        sys.exit(1)
    else:
        print("All seems right!")
    return vorovol

def histogram_volume_per_cell(vorovol,step, mask_pos,autonbins=True,nbins=20,fname=""):
    """
    This function takes info from a voronoi cell volume array and outputs a file containing the bins to graph a histogram
    INPUT: vorovol, step, mask_pos, autonbins, nbins -> an array containing the volume of each voronoi cell, time step of the simulation, mask for filtering particles, a bool for letting numpy calculate the optimal bins for the histogram, if falsee specify the number of bins
    OUTPUT: post/histo_vorocells_vol{name}_{step}.txt : file containing volume bins and number of voronoi cells in that bin
    """
    auxlog=INFOPRINT("Calculating histogram volume per voronoi cell")
    if autonbins:
        Nhist,Vbin_edges=np.histogram(vorovol[mask_pos],bins='auto')
    else:
        Nhist,Vbin_edges=np.histogram(vorovol[mask_pos],bins=nbins)

    #If we want an histogram, it's easier to use matplotlib's hist, however numpy can be useful for creating a graph, in which case we can use the half-bin-value to average the pressure per particle.
    Vhalf=np.zeros(len(Vbin_edges)-1)
    for i in np.arange(len(Vbin_edges)-1):
        Vhalf[i]=0.5*(Vbin_edges[i]+Vbin_edges[i+1])
    Nhist=np.nan_to_num(Nhist)
    np.savetxt(f"post/histo_vorocells_vol{fname}_{step}.txt",np.column_stack((Vhalf,Nhist)))

def print_vorovol(vorovol,step,fname):
    """
    This function prints a file with voronoi cell volume info to be used by plt's histogram
    INPUT: vorovol, step, fname -> array containing volume of each voroni cell, timestep of the simulation and fname an additional name indicator for the file
    OUTPUT: post/vol_per_vorocell{fname}_{step}.txt : file containing the vorovol array
    """
    auxlog=INFOPRINT("Printing voronoi cell volume array to file")
    np.savetxt(f"post/vol_per_vorocell{fname}_{step}.txt",np.column_stack((vorovol)))


#------------Series computation---------------
def series(screen_factor=np.zeros(3),stride=1,only_last=False, ncmin=2, reuse_vorofile=False):
    print(f'# Innitiating postprocessing to read every {stride} files.')
    if only_last==True:
        print('# Using only last file')
    #files to write information in: packing fraction, coordination number, cundall constant.
    #No need for files respecting histograms as those are written in the respective histogram functions
    #contacts_pattern are exit files from LIGGGHTS
    contacts_pattern="out/dump-contacts_*.gz"
    packfile="post/packing_fraction.txt"
    zfile="post/z.txt"
    cundallfile="post/cundall.txt"
    with open(packfile,'w') as opfile, open(zfile,'w') as ozfile, open(cundallfile,'w') as ocufile:
        fnames=glob.glob(contacts_pattern)#A list with the names of all dump_contacts files
        #sort the list using the time step number, that is dump-contacts_350.gz should go before dump-contacts_500.gz
        #in order to do this split the name using the _ choosing the last part (350.gz) and then the . choosing the first part (350)
        fnames.sort(key=lambda x: int(x.split('_')[-1].split('.')[0]))
        ii=0
        for fname, ii in zip(fnames,range(len(fnames))):
            if ii%stride != 0 and fname != fnames[-1] and fname != fnames[0]:
                continue
            if only_last and fname != fnames[-1]:
                continue
            step=int(fname.split('_')[-1].split('.')[0])
            print(77*"#")
            print(INFO+f" Processing step: {step}\n")
            #Use try excerpt to get errors
            try:
                #read particle data, from our naming convention out/dump_*.gz contains particle information while out/dump-contacts_*.gz contains per pair contact info
                ID,TYPE,mass,x,y,z,vx,vy,vz,fx,fy,fz,r,omegax,omegay,omegaz,tqx,tqy,tqz = read_particle_data(fname.replace("-contacts",""))
                x1,y1,z1,x2,y2,z2,ID1,ID2,IDperiod,fcx,fcy,fcz,fnx,fny,fnz,ftx,fty,ftz,tx,ty,tz,cArea,delta,cx,cy,cz=read_contact_data(fname)
                #Calculate screen length
                DMEAN=r.max()+r.min()
                print(INFO+f" Using Local DMEAN={DMEAN}")
                screen = DMEAN*np.array(screen_factor)
                #filter the particles using a mask
                counts,z_avg = count_particle_contacts(ID,ID1,ID2)
                limits = get_box_boundaries(x,y,z,r, screen)
                mask_pos=get_mask(x,y,z,r,limits)#use the box limits as screen to select particles inside the box
                mask_floating = counts < ncmin #floating particles
                mask_active = counts >= ncmin #non-floating particles
                mask_pos_contacts = get_mask(cx,cy,cz, np.zeros_like(cx), limits)#All contacts within the box
                #Use intersection update to choose only the contacts where both particles are inside the box and are not floating
                mask_pos_contacts &= np.isin(ID1, ID[mask_pos & mask_active]) & np.isin(ID2, ID[mask_pos & mask_active])
                print(INFO+f" Total number of particles: {ID.size}")
                print(INFO+f" Total number of contacts: {ID1.size}")
                print(INFO+f" Number of particles filtered by position: {np.count_nonzero(mask_pos)}")
                print(INFO+f" Number of floating particles: {np.count_nonzero(mask_floating)}")
                print(INFO+f" Number of floating particles filtered by position: {np.count_nonzero(mask_pos & mask_pos)}")
                print(INFO+f" Number of contacts filtered by position: {np.count_nonzero(mask_pos_contacts)}")
                #Now filter by force magnitude, excluiding small forces and torques
                FORCEFACTOR = 5.0e-2
                if np.any(mask_pos_contacts):
                    fnorm=np.sqrt(fnx**2+fny**2+fnz**2)
                    mask_pos_contacts &= (fnorm >= fnorm.mean()*FORCEFACTOR)
                    ct=np.sqrt(tx**2+ty**2+tz**2)
                    mask_pos_contacts &= (ct >= ct.mean()*FORCEFACTOR)
                #All particles should be good for statistical analysis. So compute and write in the files
                opfile.write("{} {}\n".format(step, packing_fraction(x,y,z,r,limits,mask_pos,screen)))
                ozfile.write("{} {}\n".format(step, mean_coordination_number(mask_pos,mask_pos_contacts)))
                ocufile.write("{} {} {} \n".format(step, *cundall(fx,fy,fz,fcx,fcy,fcz,mask_pos,mask_pos_contacts)))

                #Calculate pressure histograms
                pressure_filtered=pressure_per_particle(ID, ID1, ID2, fcx,fcy,fcz,mask_pos_contacts)
                histogram_pressure_per_particle(pressure_filtered[pressure_filtered != 0.],step,fname="_filtered")
                print_pressure(pressure_filtered,step,fname="_filtered")
                pressure_total=pressure_per_particle(ID,ID1,ID2,fcx,fcy,fcz,np.ones(ID1.size,dtype=bool))
                histogram_pressure_per_particle(pressure_total,step,fname="_total")
                print_pressure(pressure_total,step,fname="_total")

                #Calculate volume histograms
                vorovol=voronoi_volume(x,y,z,r,step)
                histogram_volume_per_cell(vorovol,step,mask_pos,fname="_filtered")
                print_vorovol(vorovol[mask_pos],step,fname="_filtered")
                histogram_volume_per_cell(vorovol,step,np.ones(ID.size, dtype=bool),fname="_total")
                print_vorovol(vorovol,step,fname="_total")
                
            except Exception as e:
                print(e)
                pass

#Now call the code
#To make sure all our codes work, they should be in ./post/[filename] and ./post, to ensure we read from the folders in this direction we append ./
#sys.path.append("./")
#fname = sys.argv[1].replace(".py","")#that is filename = postproc
#config = importlib.import_module(fname)#import all the functions to this code
#call postproc
series(screen_factor=np.zeros(3),stride=10,only_last=False, ncmin=2, reuse_vorofile=False)

                
                


