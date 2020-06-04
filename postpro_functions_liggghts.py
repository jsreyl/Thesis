
import sys
import os
import numpy as np
import glob
import importlib
#from evtk.hl import pointsToVTK
from pyevtk.hl import pointsToVTK # installed using pip install pyevtk
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



# Voronoi tesseletation based on voro++
#from tess import Container # pip install --user tess

def read_particles_info(fname, version="0.1"):
    """
    This function reads a single dump file and returns the data for particles
    INPUT:
    fname: Filename to be processed
    OUTPUT:
    returns arrays with the following info (in this order):
    id type mass x y z vx vy vz fx fy fz radius diameter omegax omegay omegaz tqx tqy tqz ix iy iz xu yu
    For version >= 0.2
    xu = Unwrapped x coordinate (for periodic systems, is the actual position without being wrapped by the periodic positions)
    ix = periodic box id
    """
    tmp=INFOPRINT("read particles")
    print("# Reading dump file : " + fname + " ... ")
    if "0.1" == version:
        iddata,typedata,mass,x,y,z,vx,vy,vz,fx,fy,fz,radius,diameter,omegax,omegay,omegaz,tqx,tqy,tqz=np.loadtxt(fname, skiprows=9, unpack=True)
    elif "0.2" == version:
        iddata,typedata,mass,x,y,z,vx,vy,vz,fx,fy,fz,radius,diameter,omegax,omegay,omegaz,tqx,tqy,tqz,ix,iy,iz,xu,yu=np.loadtxt(fname, skiprows=9, unpack=True)
    print(f"# Number of particles read: {iddata.size}")
    print("# iddata has been shifted by -1 to make it start at 0")
    iddata = iddata.astype(np.int32)
    typedata = typedata.astype(np.int32)
    iddata = iddata-1
    # sort data since ids are not in order
    idxarray = np.argsort(iddata)
    if "0.1" == version:
        return iddata[idxarray],typedata[idxarray],mass[idxarray],x[idxarray],y[idxarray],z[idxarray],vx[idxarray],vy[idxarray],vz[idxarray],fx[idxarray],fy[idxarray],fz[idxarray],radius[idxarray],diameter[idxarray],omegax[idxarray],omegay[idxarray],omegaz[idxarray],tqx[idxarray],tqy[idxarray],tqz[idxarray]
    elif "0.2" == version:
        return iddata[idxarray],typedata[idxarray],mass[idxarray],x[idxarray],y[idxarray],z[idxarray],vx[idxarray],vy[idxarray],vz[idxarray],fx[idxarray],fy[idxarray],fz[idxarray],radius[idxarray],diameter[idxarray],omegax[idxarray],omegay[idxarray],omegaz[idxarray],tqx[idxarray],tqy[idxarray],tqz[idxarray],ix[idxarray],iy[idxarray],iz[idxarray],xu[idxarray],yu[idxarray]

def read_contacts_info(fname):
    """
    This function reads a single dump file and returns the data for contacts
    INPUT:
    fname: Filename to be processed
    OUTPUT:
    returns arrays with the following info (in this order):
    # What is printed: pos id force force_normal force_tangential torque torque_normal torque_tangential contactArea delta contactPoint
    pos1x pos1y pos1z pos2x pos2y pos2z id1 id2 idperiodic fx fy fz fnorm fnx fny fnz ftx fty ftz tx ty tz tnx tny tnz ttx tty ttz area delta cx cy cz
    """
    auxlog=INFOPRINT("read contacts")
    print("# Reading dump file : " + fname + " ... ")
    pos1x,pos1y,pos1z,pos2x,pos2y,pos2z,id1,id2,idperiodic,fx,fy,fz,fnorm,fnx,fny,fnz,ftx,fty,ftz,tx,ty,tz,tnx,tny,tnz,ttx,tty,ttz,area,delta,cx,cy,cz=np.loadtxt(fname, skiprows=9, unpack=True)
    print(f"# Number of contacts read: {pos1x.size}")
    print("# id has been shifted by -1 to make it start at 0")
    id1=id1.astype(np.int32)
    id2=id2.astype(np.int32)
    id1 = id1-1
    id2 = id2-1
    return pos1x,pos1y,pos1z,pos2x,pos2y,pos2z,id1,id2,idperiodic,fx,fy,fz,fnorm,fnx,fny,fnz,ftx,fty,ftz,tx,ty,tz,tnx,tny,tnz,ttx,tty,ttz,area,delta,cx,cy,cz

def read_walls_info(fname):
    """
    This function reads a single csv file and returns the data for walls
    INPUT:
    fname: Filename to be processed
    OUTPUT:
    returns arrays with the following info (in this order):
    t,wallxmin,wallxmax,wallymin,wallymax,wallzmin,wallzmax,fwx1,fwx2,fwy1,fwy2,fwz1,fwz2
    """
    print("# Reading csv file : " + fname + " ... ")
    wallsdata = np.loadtxt(fname, skiprows=1, unpack=True, delimiter=',')
    return wallsdata
    #t,wallxmin,wallxmax,wallymin,wallymax,wallzmin,wallzmax,fwx1,fwx2,fwy1,fwy2,fwz1,fwz2=np.loadtxt(fname, skiprows=1, unpack=True, delimiter=',')
    #return t,wallxmin,wallxmax,wallymin,wallymax,wallzmin,wallzmax,fwx1,fwx2,fwy1,fwy2,fwz1,fwz2

def count_particle_contacts(iddata, id1, id2):
    """
    This function returns the number of contacts per particles
    NP   : Number of particles
    id1/2: List with ids per contact
    """
    auxlog=INFOPRINT("Counting contacts per particle ")
    # count the number of contacts per particle id with at least one contact
    uid, counts = np.unique(np.concatenate((id1, id2)), return_counts=True)
    # Now add the particles with no contacts, and put their count to zero
    # In the future maybe check: https://stackoverflow.com/questions/8251541/numpy-for-every-element-in-one-array-find-the-index-in-another-array
    full_counts = np.zeros_like(iddata)
    #for ii in np.arange(uid.size):
    #    idx = np.where(uid[ii] == iddata)
    for ii, idval in enumerate(uid):
        idx = np.where(iddata == idval)
        if idx[0].size != 1:
            print(ERROR + " Multiple indexes or no index found")
            print(ii)
            print(np.count_nonzero(idx))
            print(idx)
            print(idval)
            sys.exit(1)
        full_counts[idx] = counts[ii]
    print(INFO + f"full_counts.size (equal to number of particles): {full_counts.size}")
    print(INFO + f"sum(full_counts)/2 (equal to number of contacts ): {int(full_counts.sum()/2)}")
    if int(full_counts.sum()/2 != id1.size) or int(full_counts.sum()/2 != id2.size):
        print(ERROR + "sum of contacts counts does not equal number of contacts")
        print(f"id1.size = {id1.size}")
        print(f"id2.size = {id2.size}")
    return full_counts

def packing_fraction_snapshot(x, y, z, rad, limits, mask_pos, screen=np.zeros(3)):
    auxlog = INFOPRINT("Computing packing fraction from rectangular coordinates ")
    #limits=get_boundary_limits(x, y, z, rad, screen)
    VT=(limits[1]-limits[0])*(limits[3]-limits[2])*(limits[5]-limits[4])
    VP=4.0*np.pi*np.sum(np.power(rad[mask_pos], 3))/3.0
    #print("# Total     particles: {}".format(len(x)))
    #print("# Processed particles: {}".format(len(x[mask])))
    return VP/VT

def packing_fraction_voronoi_snapshot(x, y, z, rad, limits, mask_pos, screen=np.zeros(3), reuse_vorofile=False):
    auxlog = INFOPRINT("Computing packing fraction Voronoi by calling voro++ ")
    # get packing limits without any screen.
    # Ignore later the particles close to the wall
    # (they have negative index neighbors after the tessellation
    BNAME="ixyzr.txt"
    voro_file_exists = os.path.isfile(BNAME + ".vol")
    if voro_file_exists and reuse_vorofile:
        print(WARNING + "Reading voro data from already existing file : " + BNAME + ".vol")
    else:
        print("# Saving coords")
        np.savetxt(BNAME, np.column_stack((np.arange(len(x), dtype=np.int64), x, y, z, rad)), fmt="%i %.18e %.18e %.18e %.18e")
        args=" -r -v -o " + str((x-rad).min()) + " " + str((x+rad).max()) + " " + str((y-rad).min()) + " " + str((y+rad).max()) + " " + str((z-rad).min()) + " " + str((z+rad).max()) + " " + BNAME
        print("# Calling voro++ (this might take up to a couple of minutes ...) ")
        print(f"# args for voro++: {args}")
        os.system("voro++ " + args)

    # load the voronoi data
    print("# \tLoading and filtering voronoi data ...")
    IFNAME=BNAME + ".vol"
    ii,xdata,ydata,zdata,vorovol,raddata = np.loadtxt(IFNAME, unpack=True)
    print("# Checking voro data ...")
    if xdata.size != x.size :
        print(ERROR + "total data read is different from original data")
        print(ERROR + f"x.size:     {x.size}")
        print(ERROR + f"xvoro.size: {xdata.size}")
        sys.exit(1)
    if not np.allclose(rad, raddata):
        print(ERROR + "Rad data printed and read are different")
        sys.exit(1)
    print("# Computing and returning nu ...")
    nu=np.pi*4*np.sum(raddata[mask_pos]**3)/np.sum(vorovol[mask_pos])/3.0
    #print(nu)
    #print(4*np.pi*(raddata[mask_pos]**3)/vorovol[mask_pos]/3)
    #print(4*np.pi*((raddata[mask_pos]**3)/vorovol[mask_pos]/3).mean())
    #input()
    return nu

#def mean_coordination_number_snapshot(x, y, z, rad, cx, cy, cz, screen=np.zeros(3)):
def mean_coordination_number_snapshot(mask_pos, mask_pos_contacts):
    auxlog = INFOPRINT("Computing mean coordination number ")
    NP = np.count_nonzero(mask_pos)
    NC = np.count_nonzero(mask_pos_contacts)
    return 2*NC/NP

def active_mean_coordination_number_snapshot(mask_pos, mask_pos_contacts, mask_floating):
    auxlog = INFOPRINT("Computing active mean coordination number ")
    NP = np.count_nonzero(mask_pos)
    NF = np.count_nonzero(mask_pos & mask_floating)
    NC = np.count_nonzero(mask_pos_contacts)
    if NF > NP:
        print(ERROR + f"# NP > NF\n# NP:{NP}\n# NF:{NF}")
        assert NP>NF
    if NF == NP:
        print(WARNING + f"# NP == NF\n# NP:{NP}\n# NF:{NF}")
        return 0
    return 2*NC/(NP-NF)


def proportion_floating_particles_snapshot(mask_pos, mask_floating):
    auxlog = INFOPRINT("Proportion floating particles snapshot ")
    return np.count_nonzero(mask_pos & mask_floating)/np.count_nonzero(mask_pos)

def beta(rad, counts, mask_pos, mask_floating):
    """
    This function computes the relative importance of the floating particles in terms of size or volume
    """
    auxlog = INFOPRINT("Computing beta ")
    r_floating_mean = rad[mask_pos & mask_floating].mean()
    r_mean = rad[mask_pos].mean()
    v_floating = np.power(rad[mask_pos & mask_floating], 3).sum()
    v_all = np.power(rad[mask_pos], 3).sum()
    return r_floating_mean/r_mean, v_floating/v_all

def cundall(fx, fy, fz, cfx, cfy, cfz, mask_pos, mask_pos_contacts):
    """
    Cundall is defined as sum F_net,particle/ sum |F_contact|
    Ideally it must be zero for a mechanically stable state
    """
    auxlog = INFOPRINT("Computing cundall parameter ")
    cundall_filtered = 0.0
    if np.any(mask_pos_contacts):
        cundall_filtered = np.sum(np.sqrt(fx[mask_pos]**2 + fy[mask_pos]**2 + fz[mask_pos]**2))/np.sum(np.sqrt(cfx[mask_pos_contacts]**2 + cfy[mask_pos_contacts]**2 + cfz[mask_pos_contacts]**2))
    cundall_full = np.sum(np.sqrt(fx**2 + fy**2 + fz**2))/np.sum(np.sqrt(cfx**2 + cfy**2 + cfz**2))
    return cundall_full, cundall_filtered

def printvtk(iddata, x, y, z, rad, vx, vy, vz, fx, fy, fz, omegax, omegay, omegaz, tx, ty, tz, counts, mask_pos, mask_floating, ii):
    auxlog = INFOPRINT("Printing to vtk ")
    pointsToVTK(f"post/particles-{ii}", x, y, z, data={"id":iddata+1, "idnew":iddata, "radius":rad, "counts":counts,"mask_pos":mask_pos.astype(np.int32), "mask_floating":mask_floating.astype(np.int32),
                                                       "vx":vx, "vy":vy, "vz":vz, "fx":fx, "fy":fy, "fz":fz, "omegax":omegax, "omegay":omegay, "omegaz":omegaz, "tx":tx, "ty":ty, "tz":tz})

def histogram_force_per_size(fx, fy, fz, rad, mask_pos, mask_floating, step):
    auxlog = INFOPRINT("Computing histogram force per size ")
    nbins = 10
    dr = (rad[mask_pos].max() - rad[mask_pos].min())/(nbins)
    drmean = (rad[mask_pos].max() + rad[mask_pos].min())/2
    fnorm = np.sqrt(fx**2 + fy**2 + fz**2)
    radbin = ((rad-rad.min())/dr).astype(np.int32)
    radbin[radbin>=nbins]=nbins-1
    histo = np.zeros(nbins)
    count = np.zeros_like(histo)
    for rbin,f in zip(radbin[mask_pos], fnorm[mask_pos]):
        histo[rbin] += f
        count[rbin] += 1
    favg = fnorm[mask_pos].mean()
    histo = np.nan_to_num(histo/count/favg)
    np.savetxt(f"post/histo_rad_f-{step}.txt", np.column_stack((dr*np.arange(nbins)/drmean, histo)))

def histogram_nc_per_size(counts, rad, mask_pos, mask_floating, step):
    auxlog = INFOPRINT("Computing histogram nc per size ")
    nbins = 20
    dr = (rad.max() - rad.min())/(nbins)
    drmean = (rad.max() + rad.min())/2
    radbin = ((rad-rad.min())/dr).astype(np.int32)
    radbin[radbin>=nbins]=nbins-1
    histo = np.zeros(nbins)
    count = np.zeros_like(histo)
    for rbin, val in zip(radbin[mask_pos], counts[mask_pos]):
        histo[rbin] += val
        count[rbin] += 1
    histo = np.nan_to_num(histo/count)
    np.savetxt(f"post/histo_rad_nc-{step}.txt", np.column_stack((dr*np.arange(nbins)/drmean, histo)))

def shear_profile(x, y, z, xu, yu, x0, y0, z0, rad, mask_pos, mask_floating, step, nbins=20):
    auxlog = INFOPRINT("Computing shear profile ")
    # TODO: Filter particles by mask_pos
    # TODO: Normalize by wall z pos: Not easy , that info is not printed at the same time
    # TODO: Normalize by wall velocity: Not easy, that info is only printed on vtk files, not at the same time, better to extract it from config.servo_simlpeshear
    zi = (z - rad).min()
    zf = (z + rad).max()
    dz = (zf - zi)/(nbins)
    dxmean = np.zeros(nbins)
    dymean = np.zeros(nbins)
    dzmean = np.zeros(nbins)
    counter = np.zeros(nbins, dtype=np.int)
    for ii in range(len(x)):
        izbin = int((z[ii]-zi)/dz)
        dxmean[izbin] += xu[ii] - x0[ii]
        dymean[izbin] += yu[ii] - y0[ii]
        dzmean[izbin] +=  z[ii] - z0[ii]
        counter[izbin] += 1
    with np.errstate(divide='ignore',invalid='ignore'):
        dxmean /= counter
        dymean /= counter
        dzmean /= counter
    #dxmean /= dxmean.max()
    #dymean /= dymean.max()
    #dzmean /= dzmean.max()
    #(np.linspace(zi, zf, nbins)
    np.savetxt(f"post/shear_profile-{step}.txt", np.column_stack((zi + dz*range(nbins), dxmean, dymean, dzmean,
                                                                  np.linspace(0, 1, nbins), dxmean/dxmean.max(),
                                                                  dymean/dymean.max(), dzmean/dzmean.max())))
    # Voronoi tesseletation based on voro++
#from tess import Container # pip install --user tess

def get_boundary_limits(x, y, z, rad, screen):
    auxlog=INFOPRINT("Computing boundary limits ")
    # compute min/max lenght around each direction
    limits=np.zeros(6)
    limits[0]=np.min(x-rad)+screen[0]
    limits[1]=np.max(x+rad)-screen[0]
    limits[2]=np.min(y-rad)+screen[1]
    limits[3]=np.max(y+rad)-screen[1]
    limits[4]=np.min(z-rad)+screen[2]
    limits[5]=np.max(z+rad)-screen[2]
    return limits

def get_mask(x, y, z, radius, limits):
    auxlog=INFOPRINT("Computing position masks ")
    # compute mask for particles/contacts inside limits
    masks=[x>=limits[0], x<=limits[1], y>=limits[2], y<=limits[3], z>=limits[4], z<=limits[5]]
    #masks=[x-radius>=limits[0], x+radius<=limits[1], y-radius>=limits[2], y+radius<=limits[3], z-radius>=limits[4], z+radius<=limits[5]]
    mask=masks[0] & masks[1] & masks[2] & masks[3] & masks[4] & masks[5]
    return mask

def histogram_nc(counts, mask_pos, mask_floating, step):
    auxlog = INFOPRINT("Computing histogram nc ")
    bins = np.arange(0, 20)
    histo, bin_edges = np.histogram(counts[mask_pos], bins=bins, density=True)
    np.savetxt(f"post/histo_nc-{step}.txt", np.column_stack((bin_edges[:-1], histo)))

def series(screen_factor=np.zeros(3), stride=1, only_last=False, ncmin=2, reuse_vorofile=False, walls_info_file="out/wallslog.csv", version="0.1"):
    # get files inside dname
    print("# Creating step-time series with stride: {}".format(stride))
    print("# First and last files always included")
    # read walls info
    #t,wallxmin,wallxmax,wallymin,wallymax,wallzmin,wallzmax,fwx1,fwx2,fwy1,fwy2,fwz1,fwz2=read_walls_info(walls_info_file)
    wallsdata = read_walls_info(walls_info_file)
    # check if read only last
    if True==only_last:
        print("# Using only last file")
    # set filenames    
    contacts_pattern="out/dump-contacts_*.gz"
    packfile="post/packing_fraction.txt"
    packvorofile="post/packing_fraction_voronoi.txt"
    zfile="post/z.txt"
    azfile="post/az.txt" # active coordination
    kappafile="post/kappa.txt"
    beta1fname="post/beta1.txt"
    beta2fname="post/beta2.txt"
    cundallfname="post/cundall.txt"
    with open(packfile, "w") as opfile, open(packvorofile, "w") as ovfile, open(zfile, "w") as ozfile, open(azfile, "w") as oazfile, open(kappafile, "w") as okfile, open(beta1fname, "w") as ob1file, open(beta2fname, "w") as ob2file, open(cundallfname, "w") as ocufile:
        fnames=glob.glob(contacts_pattern)
        fnames.sort(key=lambda s: int(s.split('_')[-1].split('.')[0]))
        ii=0
        if "0.2" == version:
            iddata,typedata,mass,x0,y0,z0,vx,vy,vz,fx,fy,fz,radius,diameter,omegax,omegay,omegaz,tqx,tqy,tqz,ix,iy,iz,xu,yu = read_particles_info(fnames[0].replace("-contacts", ""), version)
        for fname, ii in zip(fnames, range(len(fnames))):
            if ii%stride != 0 and fname != fnames[-1] and fname != fnames[0]:
                continue
            if only_last and fname != fnames[-1]:
                continue
            step=int(fname.split('_')[-1].split('.')[0])
            print(77*"#")
            print(INFO + f" Processing step : {step}\n")
            try:
                if "0.1" == version:
                    iddata,typedata,mass,x,y,z,vx,vy,vz,fx,fy,fz,radius,diameter,omegax,omegay,omegaz,tqx,tqy,tqz = read_particles_info(fname.replace("-contacts", ""), version)
                elif "0.2" == version:
                    iddata,typedata,mass,x,y,z,vx,vy,vz,fx,fy,fz,radius,diameter,omegax,omegay,omegaz,tqx,tqy,tqz,ix,iy,iz,xu,yu = read_particles_info(fname.replace("-contacts", ""), version)
                pos1x,pos1y,pos1z,pos2x,pos2y,pos2z,id1,id2,idperiodic,cfx,cfy,cfz,fnorm,fnx,fny,fnz,ftx,fty,ftz,tx,ty,tz,tnx,tny,tnz,ttx,tty,ttz,area,delta,cx,cy,cz = read_contacts_info(fname)
                # compute the screen lenght
                #DMEAN=0.006 # this changes if the system expanded!
                DMEAN = 0.5*(diameter.max() + diameter.min())
                print(INFO + f" LOCALDMEAN USED: {DMEAN}")
                screen = DMEAN*np.array(screen_factor)
                # create some utility mask to filter particles
                counts = count_particle_contacts(iddata, id1, id2)
                limits = get_boundary_limits(x, y, z, radius, screen)
                mask_pos = get_mask(x, y, z, radius, limits) # particles inside screen length
                mask_floating = counts < ncmin # floating particles
                mask_active  = counts >= ncmin # non-floating particles
                mask_pos_contacts = get_mask(cx, cy, cz, np.zeros_like(cx),limits) # contacts inside screen : allows contacts with outside particles
                #mask_pos_contacts &= np.isin(id1, iddata[mask_pos]) & np.isin(id2, iddata[mask_pos]) & mask_pos_contacts  # both particles inside domain
                mask_pos_contacts &= np.isin(id1, iddata[mask_pos & mask_active]) & np.isin(id2, iddata[mask_pos & mask_active]) # both particles inside domain, and active
                print(INFO + f"Total number of particles: {iddata.size}")
                print(INFO + f"Total number of contacts : {cx.size}")
                print(INFO + f"Number particles filtered by position: {np.count_nonzero(mask_pos)}")
                print(INFO + f"Number of floating particles: {np.count_nonzero(mask_floating)}")
                print(INFO + f"Number of floating particles, filtered by position: {np.count_nonzero(mask_floating & mask_pos)}")
                print(INFO + f"Number of contacts filtered by position: {np.count_nonzero(mask_pos_contacts)}")
                # Filter by force magnitud, excluding small forces and torques
                FORCEFACTOR = 5.0e-2
                if np.any(mask_pos_contacts):
                    mask_pos_contacts &= (fnorm >= fnorm[mask_pos_contacts].mean()*FORCEFACTOR)
                    ct = np.sqrt(tx**2 + ty**2 + tz**2)
                    mask_pos_contacts &= (ct >= ct[mask_pos_contacts].mean()*FORCEFACTOR)
                # Now compute and write
                opfile.write("{}   {}\n".format(step, packing_fraction_snapshot(x, y, z, radius, limits, mask_pos, screen)))
                ovfile.write("{}   {}\n".format(step, packing_fraction_voronoi_snapshot(x, y, z, radius, limits, mask_pos, screen, reuse_vorofile)))
                ozfile.write("{}   {}\n".format(step, mean_coordination_number_snapshot(mask_pos, mask_pos_contacts)))
                oazfile.write("{}   {}\n".format(step, active_mean_coordination_number_snapshot(mask_pos, mask_pos_contacts, mask_floating)))
                okfile.write("{}   {}\n".format(step, proportion_floating_particles_snapshot(mask_pos, mask_floating)))
                beta1, beta2 = beta(radius, counts, mask_pos, mask_floating)
                ob1file.write("{}   {} \n".format(step, beta1))
                ob2file.write("{}   {} \n".format(step, beta2))
                ocufile.write("{}   {}  {} \n".format(step, *cundall(fx, fy, fz, cfx, cfy, cfz, mask_pos, mask_pos_contacts)))
                printvtk(iddata, x, y, z, radius, vx, vy, vz, fx, fy, fz, omegax, omegay, omegaz, tqx, tqy, tqz, counts, mask_pos, mask_floating, step)
                histogram_force_per_size(fx, fy, fz, radius, mask_pos, mask_floating, step)
                histogram_nc_per_size(counts, radius, mask_pos, mask_floating, step)
                histogram_nc(counts, mask_pos, mask_floating, step)
                if "0.2" == version:
                    shear_profile(x, y, z, xu, yu, x0, y0, z0, radius, mask_pos, mask_floating, step, nbins=10)
            except Exception as e:
                print(e)
                pass

# Testing area
#iddata,typedata,mass,x,y,z,vx,vy,vz,fx,fy,fz,radius,diameter,omegax,omegay,omegaz,tqx,tqy,tqz = read_particles_info(sys.argv[1])
#pos1x,pos1y,pos1z,pos2x,pos2y,pos2z,id1,id2,idperiodic,cfx,cfy,cfz,fnorm,fnx,fny,fnz,ftx,fty,ftz,tx,ty,tz,tnx,tny,tnz,ttx,tty,ttz,area,delta,cx,cy,cz = read_contacts_info(sys.argv[2])
#print(packing_fraction_snapshot(x, y, z, radius, [0.0065, 0.0065, 0.0065]))
#print(mean_cordination_number_snapshot(x, y, z, radius, cx, cy, cz, [0.0065, 0.0065, 0.0065]))
#print(proportion_floating_particles_snapshot(x, y, z, id1, id2, ncmin=3))

# read configuration file and import it
#import config_postpro as config
sys.path.append("./")
fname = sys.argv[1].replace(".py","")
#from config_postpro import *
config = importlib.import_module(fname)
# call postprocessing
series(screen_factor=[config.SCREEN_FACTOR, config.SCREEN_FACTOR, config.SCREEN_FACTOR],
       stride=config.STRIDE, only_last=config.ONLY_LAST, ncmin=config.NCMIN,
       reuse_vorofile = config.REUSE_VOROFILE, walls_info_file=config.WALLS_INFO_FILE, version=config.POSTPRO_VERSION)
