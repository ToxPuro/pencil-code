'''
    Copyright (C) 2014-2022, Johannes Pekkilae, Miikka Vaeisalae.

    This file is part of Astaroth.

    Astaroth is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    Astaroth is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with Astaroth.  If not, see <http://www.gnu.org/licenses/>.
'''

# The following cripts are based on what was originally published in 
# Väisälä M.S., Pekkilä J., Käpylä M.J., Rheinhardt M., Shang H., 
# Krasnopolsky R., 2021, Astrophysical Journal, 907, 83. 
# doi:10.3847/1538-4357/abceca

import astar.data as ad
import astar.visual as vis
import pylab as plt 
import numpy as np 
import sys
import matplotlib as mpl

import pandas as pd 

import os 

import tqdm

import scipy.stats as stats

df_archive = pd.DataFrame()


print("sys.argv", sys.argv)

if 'mydatadir' in sys.argv:
    basicdir = "/my/directory/for/runs/"
    meshdirs = [basicdir + "run1/",
                basicdir + "run2/",
                basicdir + "run3/",
                basicdir + "run4/"]

    rlabel = "mydatadir"

    noheltext = ""


#Testmesh for powerspectral  
def testmesh(resolution, dx, dy, dz):
    xx  = np.arange(resolution, dtype=np.float64)*dx  
    yy  = np.arange(resolution, dtype=np.float64)*dy   
    zz  = np.arange(resolution, dtype=np.float64)*dz 
    xx, yy, zz = np.meshgrid(xx, yy, zz, indexing="ij")

    vecx = np.sin(2.0*xx) + np.sin(10.0*xx) + np.cos(7.0*yy)  +    np.cos(5.0*zz)
    vecy = np.sin(5.0*yy) + np.sin( 3.0*yy) + np.cos(4.0*xx)  +    np.cos( 6.0*zz)
    vecz = np.sin(6.0*zz) + np.sin( 8.0*zz) + np.cos(20.0*xx) + 10*np.cos( 9.0*yy)

    vecx = np.cos(10.0*yy) 
    vecy = 0.0*vecy 
    vecz = 0.0*vecz

    mymesh = (vecx, vecy, vecz)
    
    return mymesh 

#Calculate xy-plen averages of the system 
def xyavers(mesh, plane = 'xy'):

    if plane == 'xy':
        Bx_plane = np.mean(mesh.bb[0], axis=(0,1))
        By_plane = np.mean(mesh.bb[1], axis=(0,1))
        Bz_plane = np.mean(mesh.bb[2], axis=(0,1))
    elif plane == 'yz':
        Bx_plane = np.mean(mesh.bb[0], axis=(1,2))
        By_plane = np.mean(mesh.bb[1], axis=(1,2))
        Bz_plane = np.mean(mesh.bb[2], axis=(1,2))
    elif plane == 'xz':
        Bx_plane = np.mean(mesh.bb[0], axis=(0,2))
        By_plane = np.mean(mesh.bb[1], axis=(0,2))
        Bz_plane = np.mean(mesh.bb[2], axis=(0,2))

    Bx_plane = Bx_plane[3:-3]
    By_plane = By_plane[3:-3]
    Bz_plane = Bz_plane[3:-3]

    Bplane = (Bx_plane, By_plane, Bz_plane) 

    return Bplane

# Powerspectra calculator 
def pfft_shell(mymesh, dx, dy, dz, title = "", ffttest=False):
    freqs_x = 2.0*np.pi*np.fft.fftfreq(mymesh.shape[0], dx)
    freqs_y = 2.0*np.pi*np.fft.fftfreq(mymesh.shape[1], dy)
    freqs_z = 2.0*np.pi*np.fft.fftfreq(mymesh.shape[2], dz)

    NX = mymesh.shape[0]
    NY = mymesh.shape[1]
    NZ = mymesh.shape[2]

    #3D Fouries transform
    fft_mesh = np.fft.fftn(mymesh)

    kkx, kky, kkz = np.meshgrid(freqs_x, freqs_y, freqs_z, indexing='ij')

    k_arr  = np.arange(freqs_x.size/2)
    comp_eks    = np.zeros_like(k_arr)
    comp_nshell = np.zeros_like(k_arr)

    if ffttest:
        #plt.pcolormesh(kkx[32,:,:], kky[32,:,:], mymesh[32,:,:])
        #plt.figure()
        #plt.pcolormesh(kky[32,:,:], kkz[32,:,:], np.abs(fft_mesh[32,:,:]))
        #plt.pcolormesh(np.abs(fft_mesh[32,:,:]))
        #plt.colorbar()
        #plt.figure()
        #plt.pcolormesh(kkx[:,32,:], kkz[:,32,:], np.abs(fft_mesh[:,32,:]))
        plt.figure()
        plt.pcolormesh(np.abs(fft_mesh[:,32,:]))
        plt.colorbar()
        plt.figure()
        plt.pcolormesh(np.real(fft_mesh[:,32,:]))
        plt.colorbar()
        plt.figure()
        plt.pcolormesh(np.imag(fft_mesh[:,32,:]))
        plt.colorbar()
        #plt.figure()
        #plt.pcolormesh(kkx[:,:,32], kky[:,:,32], np.abs(fft_mesh[:,:,32]))
        #plt.pcolormesh(np.abs(fft_mesh[:,:,32]))
        #plt.colorbar()
        plt.show()
        plt.figure()

    pbar = tqdm.tqdm(range(fft_mesh.shape[0]))
    pile = np.array([])
    for i in pbar:
        pbar.set_description("Computing prowerspectra %s" % title)
        for j in range(fft_mesh.shape[1]):
            for k in range(fft_mesh.shape[2]):
                comp_kx   = kkx[i, j, k]
                comp_ky   = kky[i, j, k]
                comp_kz   = kkz[i, j, k]
                comp_kabs = np.sqrt(comp_kx**2.0 + comp_ky**2.0 + comp_kz**2.0)
                comp_ik   = int(round(comp_kabs))
                comp_vfft = np.abs(fft_mesh[i,j,k])
                #Perform normalization step. 
                comp_vfft = (1.0/(NX*NY*NZ))*comp_vfft
                if ffttest:
                     if comp_vfft > 0.0:
                          print(comp_kx, comp_ky, comp_kz, comp_kabs, comp_ik, comp_vfft, fft_mesh[i,j,k])
                if comp_ik < k_arr.size: 
                    comp_eks[comp_ik]    = comp_eks[comp_ik] + comp_vfft**2.0 
                    comp_nshell[comp_ik] = comp_nshell[comp_ik] + 1 

    # This way relates to magnetic and kinetic energy. 
    comp_eks = 0.5*comp_eks

    sys.stdout.flush() 
    
    return k_arr, comp_eks, comp_nshell 

#Gather powerspectra calculations results. 
def powerspectra_shell(mesh, dx, dy, dz, uu=False, ffttest=False):
    if uu:
        meshx = mesh.uu[0][3:-3, 3:-3, 3:-3]
        meshy = mesh.uu[1][3:-3, 3:-3, 3:-3]
        meshz = mesh.uu[2][3:-3, 3:-3, 3:-3]
    elif ffttest:
        mymesh = testmesh(mesh.minfo.contents['AC_nx'], dx, dy, dz)
        meshx = mymesh[0] 
        meshy = mymesh[1] 
        meshz = mymesh[2] 
    else:
        mesh.Bfield()
        meshx = mesh.bb[0][3:-3, 3:-3, 3:-3]
        meshy = mesh.bb[1][3:-3, 3:-3, 3:-3]
        meshz = mesh.bb[2][3:-3, 3:-3, 3:-3]

    k_arrx, eksx, nshellx = pfft_shell(meshx, dx, dy, dz, title='Ex', ffttest=ffttest)
    k_arry, eksy, nshelly = pfft_shell(meshy, dx, dy, dz, title='Ey')
    k_arrz, eksz, nshellz = pfft_shell(meshz, dx, dy, dz, title='Ez')
                
    eks_tot = eksx + eksy + eksz

    return eks_tot, eksx, k_arrx, eksy, k_arry, eksz, k_arrz

# Read reduced powerspectra 
def read_powerspectra(psfile):
    powerfile = open(psfile, "r") 
    lines = powerfile.read()
    #line = line.split("[")
    #print(lines)

    lines = lines.splitlines()
    if len(lines) < 3:
        return None, None, None, None

    # Parse system information
    sys_vars = lines[0].split(',')
    power_info = {}
    for var in sys_vars:
       temp = var.split()
       power_info[temp[0]] = float(temp[1])
    #print(power_info)
   
    # Parse frequency axis 
    freqs = lines[1].split('[')
    freqs = freqs[1]
    freqs = freqs.replace(']', '') 
    freqs = np.fromstring(freqs, sep=" ")
    #print(freqs)

    timeline  = np.array([])
    all_spect = np.zeros_like(freqs)

    for line in lines[2:]:
        line = line.split('[')
        time = np.float(line[0])
        power = line[1]
        power = power.replace(']', '') 
        power = np.fromstring(power, sep=" ")
        #print(time)
        #print(power)
        timeline = np.append(timeline, time)
        all_spect = np.vstack((all_spect, power))

    all_spect = all_spect[1:, :]

    #print(all_spect)
    #print(timeline)        

    powerfile.close()

    return power_info, timeline, freqs, all_spect 

# Get energy spectra and write it to a file. 
def extract_spectra(meshdirs, uu=False, noheltext=""):
    for meshdir in meshdirs:

        mesh_file_numbers = ad.read.parse_directory(meshdir)
     
        if len(mesh_file_numbers) > 0: 
            maxfiles = np.amax(mesh_file_numbers)
    
            #Get text for hearder 
            mesh = ad.read.Mesh(0, fdir=meshdir, only_info = True)
            resolution = mesh.minfo.contents['AC_nx'      ]
            nu         = mesh.minfo.contents['AC_nu_visc' ]
            eta        = mesh.minfo.contents['AC_eta']
            dsx        = mesh.minfo.contents['AC_dsx']
            dsy        = mesh.minfo.contents['AC_dsy']
            dsz        = mesh.minfo.contents['AC_dsz']
            relhel     = mesh.minfo.contents['AC_relhel']
            Prandtl    = nu/eta
            kk         = (mesh.minfo.contents['AC_kmax']+mesh.minfo.contents['AC_kmin'])/2.0 
            headertext = r"%i$^3$, $\eta$ = %.2e, $\sigma$ = %.0f" % (resolution, eta, relhel)
            filename   = "%i_k%.0f_eta%.2e.esp" % (resolution, kk, eta)

            if uu:
                vartype="_uu_"
            else:
                vartype="_bb_"

            writefreq=True

            #for i in mesh_file_numbers:
            pbar = tqdm.tqdm(mesh_file_numbers)
            #pbar = tqdm.tqdm(mesh_file_numbers[:15]) #FOR TEST 
            print("pow_etot" + filename)
            etotfile = open(noheltext + "pow_etot" + vartype + filename, "w") 
            exfile   = open(noheltext + "pow_ex"   + vartype + filename, "w") 
            eyfile   = open(noheltext + "pow_ey"   + vartype + filename, "w") 
            ezfile   = open(noheltext + "pow_ez"   + vartype + filename, "w")
            headerinfo = "nx %e, dsx %e, dsy %e, dsz %e, relhel %e, kaver %e, eta %e, nu %e \n" % (resolution, dsx, dsy, dsz, relhel, kk, eta, nu) 
            etotfile.write(headerinfo)
            exfile.write(  headerinfo) 
            eyfile.write(  headerinfo) 
            ezfile.write(  headerinfo) 
            for i in pbar:
                pbar.set_description("Processing %i" % i)
                mesh = ad.read.Mesh(i, fdir=meshdir) 
                #print(" %i / %i" % (i, maxfiles))
                if mesh.ok:
                    #print("TIME", mesh.timestamp)
                    Etot, Exx, freqs_x, Eyy, freqs_y, Ezz, freqs_z = powerspectra_shell(mesh, dsx, dsy, dsz, uu=uu)
           
                if writefreq:
                    etotfile.write("kk"+ " " + np.array2string(freqs_x, max_line_width = 1000000) + "\n")
                    exfile.write("kx"  + " " + np.array2string(freqs_x, max_line_width = 1000000) + "\n")
                    eyfile.write("ky"  + " " + np.array2string(freqs_y, max_line_width = 1000000) + "\n")
                    ezfile.write("kz"  + " " + np.array2string(freqs_z, max_line_width = 1000000) + "\n")
                    writefreq=False

                etotfile.write(str(mesh.timestamp) + " " + np.array2string(Etot, max_line_width = 1000000) + "\n")
                exfile.write(  str(mesh.timestamp) + " " + np.array2string(Exx,  max_line_width = 1000000) + "\n")
                eyfile.write(  str(mesh.timestamp) + " " + np.array2string(Eyy,  max_line_width = 1000000) + "\n")
                ezfile.write(  str(mesh.timestamp) + " " + np.array2string(Ezz,  max_line_width = 1000000) + "\n")
 
            etotfile.close() 
            exfile.close()
            eyfile.close()
            ezfile.close()


#Extract specra from file. 
if "ext_pow" in sys.argv:
    extract_spectra(meshdirs, uu=False, noheltext=noheltext)

if "ext_powuu" in sys.argv:
    extract_spectra(meshdirs, uu=True, noheltext=noheltext)


def parse_pspec_directory(mydir, keyword):
    dirlist = os.listdir(mydir)
    dirlist = [k for k in dirlist if keyword in k]
    for i, item in enumerate(dirlist):
        tmp = item.strip('.mesh')
        tmp = tmp.strip('VTXBUF_LNRHO')
        dirlist[i] = tmp
    dirlist.sort()
    return dirlist

#Plot powerspectra 
def plot_pspec(power_info, timeline, freqs, Espec_all, titlepart, vartype, vdir = 'tot', my_xrange = None, my_yrange = None, kazantsev = False):
    if power_info != None:  
    
        plt.figure()

        mytitle = r"$%s$, $N = %i^3$, $\eta =$ %.2e, $\sigma = %i$" % (titlepart, power_info['nx'], power_info['eta'], power_info['relhel'] )
        filename   = "sigma%i_%i_eta%.2e.pdf" % (power_info['relhel'],power_info['nx'], power_info['eta'])

        kk = power_info['kaver']

        #Becaude no normalization ytet TODO remove
        #if power_info['nx'] == 512:
        #    my_yrange = None

        step = 0 
        #Etot_all[:, 1]
        nsteps = Espec_all[:,1].size
        colors = plt.cm.inferno(np.linspace(0,1, nsteps))

        for Etot in Espec_all:
            if step == 0:
                lstyle = "dashed"
            else:
                lstyle = "solid"
            if kazantsev:
                plt.plot(freqs[1:], Etot[1:]/(freqs[1:]**(3.0/2.0)), linewidth=2)
            else:
                if step == 0:
                    #plt.scatter(kk, Etot[int(kk)], label = r"$k_\mathrm{f}$", zorder=3)
                    plt.plot([kk,kk], my_yrange, linestyle = "dotted", color="black", label = r"$k_\mathrm{f}$")
                plt.plot(freqs[1:], Etot[1:], linewidth=2, color=colors[step], linestyle = lstyle) 
                if step == 4:
                    plt.plot(freqs[1:10], (5.0*Etot[1])*freqs[1:10]**(3.0/2.0), "r--", label = r'$k^\frac{3}{2}$', linewidth=2, zorder=3)
     
                step += 1 

        if kazantsev:
            kazan = "kazan"
        else:
            kazan = ""

        plt.legend()
        
        plt.yscale("log")
        plt.xscale("log")
        plt.xlabel(r"$k/k_0$")
        plt.ylabel(r"$%s$" % titlepart)
        #plt.title(mytitle)

        print("x = ", my_xrange, "y = ", my_yrange)

        if my_xrange != None:
            plt.xlim(my_xrange)
        if my_yrange != None:
            plt.ylim(my_yrange)

        savefile = kazan + vartype + vdir + "_powerspectra_" + filename
        print("saving %s..." % (savefile))        

        plt.savefig(savefile)
        plt.close()

#Plot powerspectra from file 
if "pspec" in sys.argv:
   
    pstotfiles = parse_pspec_directory(".", "pow_etot")
    psxfiles   = parse_pspec_directory(".", "pow_ex")
    psyfiles   = parse_pspec_directory(".", "pow_ey")
    pszfiles   = parse_pspec_directory(".", "pow_ez")

    my_xrange = (0.8, 300.0)
    my_yrange = (1e-13, 6e-2)
    #my_yrange = None

    if "kazan" in sys.argv:
        kazantsev = True
        #my_xrange = None 
        #my_yrange = None 
    else:
        kazantsev = False

    for psfile in pstotfiles: 
        power_info, timeline, freqs, Etot_all = read_powerspectra(psfile)
        if "_uu_" in psfile:
            vartype = "u"
        elif "_bb_" in psfile:
            vartype = "B"
        else:
            vartype = ""

        plot_pspec(power_info, timeline, freqs, Etot_all, "E_\mathrm{%s}(k)" % (vartype), vartype, my_xrange=my_xrange, my_yrange=my_yrange, kazantsev=kazantsev)
 
    for psfile in psxfiles: 
        power_info, timeline, freqs, Etot_all = read_powerspectra(psfile)
        if "_uu_" in psfile:
            vartype = "u"
        elif "_bb_" in psfile:
            vartype = "B"
        else:
            vartype = ""
        plot_pspec(power_info, timeline, freqs, Etot_all, "E_\mathrm{%s,x}(k)" % (vartype), vartype, vdir = "x", my_xrange=my_xrange, my_yrange=my_yrange, kazantsev=kazantsev)
 
    for psfile in psyfiles: 
        power_info, timeline, freqs, Etot_all = read_powerspectra(psfile)
        if "_uu_" in psfile:
            vartype = "u"
        elif "_bb_" in psfile:
            vartype = "B"
        else:
            vartype = ""
        plot_pspec(power_info, timeline, freqs, Etot_all, "E_\mathrm{%s,y}(k)" % (vartype), vartype, vdir = "y", my_xrange=my_xrange, my_yrange=my_yrange, kazantsev=kazantsev)
 
    for psfile in pszfiles: 
        power_info, timeline, freqs, Etot_all = read_powerspectra(psfile)
        if "_uu_" in psfile:
            vartype = "u"
        elif "_bb_" in psfile:
            vartype = "B"
        else:
            vartype = ""
        plot_pspec(power_info, timeline, freqs, Etot_all, "E_\mathrm{%s,z}(k)" % (vartype), vartype, vdir = "z", my_xrange=my_xrange, my_yrange=my_yrange, kazantsev=kazantsev)


# Calculate PDF of the turbulence.  
if "histogram" in sys.argv:

    for meshdir in meshdirs:

        mesh_file_numbers = ad.read.parse_directory(meshdir)
        print(mesh_file_numbers)
        maxfiles = np.amax(mesh_file_numbers)
    
        #Get text for hearder 
        mesh = ad.read.Mesh(0, fdir=meshdir, only_info = True)
        resolution = mesh.minfo.contents['AC_nx'      ]
        nu         = mesh.minfo.contents['AC_nu_visc' ]
        eta        = mesh.minfo.contents['AC_eta']
        dx        = mesh.minfo.contents['AC_dsx']
        dy        = mesh.minfo.contents['AC_dsy']
        dz        = mesh.minfo.contents['AC_dsz']
        relhel     = mesh.minfo.contents['AC_relhel']
        Prandtl    = nu/eta
        kk         = (mesh.minfo.contents['AC_kmax']+mesh.minfo.contents['AC_kmin'])/2.0 
        headertext = r"%i$^3$, $\eta$ = %.2e, $\sigma$ = %.0f" % (resolution, eta, relhel)
        filename   = "%i_k%.0f_eta%.2e.pdf" % (resolution, kk, eta)

        for i in mesh_file_numbers:
            mesh = ad.read.Mesh(i, fdir=meshdir) 
            print(" %i / %i" % (i, maxfiles))
               
            nbins = 128
            bfield = 0.6
 
            if mesh.ok:
                mesh.Bfield()
                meshx = mesh.bb[0][3:-3, 3:-3, 3:-3]
                meshy = mesh.bb[1][3:-3, 3:-3, 3:-3]
                meshz = mesh.bb[2][3:-3, 3:-3, 3:-3]
                print("TIME", mesh.timestamp)
                #plt.figure()
                histx, bhx = np.histogram(meshx, bins = nbins, range =(-bfield,bfield))
                histy, bhy = np.histogram(meshy, bins = nbins, range =(-bfield,bfield))
                histz, bhz = np.histogram(meshz, bins = nbins, range =(-bfield,bfield))
                print(histx.shape, bhx.shape)

                histx = histx/np.sum(histx)
                histy = histy/np.sum(histy)
                histz = histz/np.sum(histz)

                for ind, hval in enumerate(histx):

                    df_archive = df_archive.append({"rundir":meshdir, "resolution":resolution, "kk":kk, "time":mesh.timestamp,  "relhel":relhel,
                                                    "eta":eta, "nu":nu, "xyz":"x", "hbin":bhx[ind], "PDF":histx[ind]},
                                                    ignore_index=True)
                    df_archive = df_archive.append({"rundir":meshdir, "resolution":resolution, "kk":kk, "time":mesh.timestamp,  "relhel":relhel,
                                                    "eta":eta, "nu":nu, "xyz":"y", "hbin":bhy[ind], "PDF":histy[ind]},
                                                    ignore_index=True)
                    df_archive = df_archive.append({"rundir":meshdir, "resolution":resolution, "kk":kk, "time":mesh.timestamp,  "relhel":relhel,
                                                    "eta":eta, "nu":nu, "xyz":"z", "hbin":bhz[ind], "PDF":histz[ind]},
                                                    ignore_index=True)

                print(df_archive)


    df_archive.to_csv("PDFs.csv", index=False)

# Get and plot xy averages 
if "xyaver" in sys.argv:

    for meshdir in meshdirs:

        mesh_file_numbers = ad.read.parse_directory(meshdir)
        mesh_file_numbers = mesh_file_numbers[1:]
        print(mesh_file_numbers)
        maxfiles = np.amax(mesh_file_numbers)

    
        #Get text for hearder 
        mesh = ad.read.Mesh(0, fdir=meshdir, only_info = True)
        resolution = mesh.minfo.contents['AC_nx'      ]
        nu         = mesh.minfo.contents['AC_nu_visc' ]
        eta        = mesh.minfo.contents['AC_eta']
        dsx        = mesh.minfo.contents['AC_dsx']
        dsy        = mesh.minfo.contents['AC_dsy']
        dsz        = mesh.minfo.contents['AC_dsz']
        relhel     = mesh.minfo.contents['AC_relhel']
        Prandtl    = nu/eta
        kk         = (mesh.minfo.contents['AC_kmax']+mesh.minfo.contents['AC_kmin'])/2.0 
        headertext = r"%i$^3$, $\eta$ = %.2e, $\sigma$ = %.0f" % (resolution, eta, relhel)
        filename   = "sigma%i_%i_k%.0f_eta%.2e.png" % (relhel, resolution, kk, eta)

        fig, axs = plt.subplots(9, 1, sharex=True, figsize=(7,10))
        Bx_xyt  = np.array([])
        By_xyt  = np.array([])
        Bz_xyt  = np.array([])
        Bx_xzt  = np.array([])
        By_xzt  = np.array([])
        Bz_xzt  = np.array([])
        Bx_yzt  = np.array([])
        By_yzt  = np.array([])
        Bz_yzt  = np.array([])
        tt    = np.array([])
        initxy = 0
        for i in mesh_file_numbers:
            mesh = ad.read.Mesh(i, fdir=meshdir) 
            print(" %i / %i" % (i, maxfiles))
            if mesh.ok:
                mesh.Bfield()
                Bxy = xyavers(mesh, plane='xy')
                Bxz = xyavers(mesh, plane='xz')
                Byz = xyavers(mesh, plane='yz')
                print("Bxy[0].shape", Bxy[0].shape)
                print("Bxy[1].shape", Bxy[1].shape)
                print("Bxy[2].shape", Bxy[2].shape)

                if initxy == 0: 
                    Bx_xyt = np.zeros_like(Bxy[0])
                    By_xyt = np.zeros_like(Bxy[1])
                    Bz_xyt = np.zeros_like(Bxy[2])
                    Bx_xzt = np.zeros_like(Bxz[0])
                    By_xzt = np.zeros_like(Bxz[1])
                    Bz_xzt = np.zeros_like(Bxz[2])
                    Bx_yzt = np.zeros_like(Byz[0])
                    By_yzt = np.zeros_like(Byz[1])
                    Bz_yzt = np.zeros_like(Byz[2])
                    initxy = 1 

                Bx_xyt = np.vstack((Bx_xyt, Bxy[0]))  
                By_xyt = np.vstack((By_xyt, Bxy[1]))  
                Bz_xyt = np.vstack((Bz_xyt, Bxy[2]))  
                Bx_xzt = np.vstack((Bx_xzt, Bxz[0]))  
                By_xzt = np.vstack((By_xzt, Bxz[1]))  
                Bz_xzt = np.vstack((Bz_xzt, Bxz[2]))  
                Bx_yzt = np.vstack((Bx_yzt, Byz[0]))  
                By_yzt = np.vstack((By_yzt, Byz[1]))  
                Bz_yzt = np.vstack((Bz_yzt, Byz[2]))  

                print("Bx_xyt.shape", Bx_xyt.shape)

                tt     = np.append(tt,   mesh.timestamp)

        ttx, xx = np.meshgrid(tt, np.arange(resolution)*dsx)
        tty, yy = np.meshgrid(tt, np.arange(resolution)*dsy)
        ttz, zz = np.meshgrid(tt, np.arange(resolution)*dsz)

        datamin = np.amax([Bx_xyt, By_xyt, Bz_xyt, Bx_xzt, By_xzt, Bz_xzt, Bx_yzt, By_yzt, Bz_yzt])
        datamax = np.amin([Bx_xyt, By_xyt, Bz_xyt, Bx_xzt, By_xzt, Bz_xzt, Bx_yzt, By_yzt, Bz_yzt])

        datarange = np.amax([np.abs(datamin), np.abs(datamax)])

        #colornorm = mpl.colors.SymLogNorm(linthresh=1e-8, vmin=-datarange, vmax=datarange)
        colornorm = mpl.colors.SymLogNorm(linthresh=1e-6, vmin=-datarange, vmax=datarange)

        image1 = axs[0].pcolormesh(ttz, zz, np.transpose(Bx_xyt[1:]), norm = colornorm, cmap=plt.get_cmap('inferno'))
        image2 = axs[1].pcolormesh(ttz, zz, np.transpose(By_xyt[1:]), norm = colornorm, cmap=plt.get_cmap('inferno'))
        image3 = axs[2].pcolormesh(ttz, zz, np.transpose(Bz_xyt[1:]), norm = colornorm, cmap=plt.get_cmap('inferno'))
        image4 = axs[3].pcolormesh(tty, yy, np.transpose(Bx_xzt[1:]), norm = colornorm, cmap=plt.get_cmap('inferno'))
        image5 = axs[4].pcolormesh(tty, yy, np.transpose(By_xzt[1:]), norm = colornorm, cmap=plt.get_cmap('inferno'))
        image6 = axs[5].pcolormesh(tty, yy, np.transpose(Bz_xzt[1:]), norm = colornorm, cmap=plt.get_cmap('inferno'))
        image7 = axs[6].pcolormesh(ttx, xx, np.transpose(Bx_yzt[1:]), norm = colornorm, cmap=plt.get_cmap('inferno'))
        image8 = axs[7].pcolormesh(ttx, xx, np.transpose(By_yzt[1:]), norm = colornorm, cmap=plt.get_cmap('inferno'))
        image9 = axs[8].pcolormesh(ttx, xx, np.transpose(Bz_yzt[1:]), norm = colornorm, cmap=plt.get_cmap('inferno'))

        images = [image1, image2, image3, image4, image5, image6, image7, image8, image9]

        axs[0].set_ylabel(r"$\langle B_x \rangle_{%s}$" % "xy")
        axs[1].set_ylabel(r"$\langle B_y \rangle_{%s}$" % "xy")
        axs[2].set_ylabel(r"$\langle B_z \rangle_{%s}$" % "xy")
        axs[3].set_ylabel(r"$\langle B_x \rangle_{%s}$" % "xz")
        axs[4].set_ylabel(r"$\langle B_y \rangle_{%s}$" % "xz")
        axs[5].set_ylabel(r"$\langle B_z \rangle_{%s}$" % "xz")
        axs[6].set_ylabel(r"$\langle B_x \rangle_{%s}$" % "yz")
        axs[7].set_ylabel(r"$\langle B_y \rangle_{%s}$" % "yz")
        axs[8].set_ylabel(r"$\langle B_z \rangle_{%s}$" % "yz")

        axs[8].set_xlabel(r"$t$")

        axs[0].set_xlim(0.0, 4000.0)
        axs[1].set_xlim(0.0, 4000.0)
        axs[2].set_xlim(0.0, 4000.0)
        axs[3].set_xlim(0.0, 4000.0)
        axs[4].set_xlim(0.0, 4000.0)
        axs[5].set_xlim(0.0, 4000.0)
        axs[6].set_xlim(0.0, 4000.0)
        axs[7].set_xlim(0.0, 4000.0)
        axs[8].set_xlim(0.0, 4000.0)

        fig.colorbar(images[0], ax=axs, orientation='horizontal', fraction=.1)

        #plt.show()

        prefix = noheltext + "xy_xz_yz_" + "aver_"  

        plt.savefig(prefix + filename)  
        plt.close(fig)

        
if 'kurtosis' in sys.argv:

    for meshdir in meshdirs:
        mesh_file_numbers = ad.read.parse_directory(meshdir)
        print(mesh_file_numbers)
        maxfiles = np.amax(mesh_file_numbers)
    
        #Get text for hearder 
        mesh = ad.read.Mesh(0, fdir=meshdir, only_info = True)
        resolution = mesh.minfo.contents['AC_nx'      ]
        nu         = mesh.minfo.contents['AC_nu_visc' ]
        eta        = mesh.minfo.contents['AC_eta']
        relhel     = mesh.minfo.contents['AC_relhel']
        Prandtl    = nu/eta
        kk         = (mesh.minfo.contents['AC_kmax']+mesh.minfo.contents['AC_kmin'])/2.0 

        print("resolution", resolution, "nu", nu, "relhel", relhel)

        pbar = tqdm.tqdm(mesh_file_numbers)
        for i in pbar:
            mesh = ad.read.Mesh(i, fdir=meshdir, pdiag = False) 
            sys.stdout.flush() 
            if mesh.ok:
                mesh.Bfield()
                kurt_bx = stats.kurtosis(mesh.bb[0], axis=None)
                kurt_by = stats.kurtosis(mesh.bb[1], axis=None)
                kurt_bz = stats.kurtosis(mesh.bb[2], axis=None)
                kurt_ux = stats.kurtosis(mesh.uu[0], axis=None)
                kurt_uy = stats.kurtosis(mesh.uu[1], axis=None)
                kurt_uz = stats.kurtosis(mesh.uu[2], axis=None)

                skew_bx = stats.skew(mesh.bb[0], axis=None)
                skew_by = stats.skew(mesh.bb[1], axis=None)
                skew_bz = stats.skew(mesh.bb[2], axis=None)
                skew_ux = stats.skew(mesh.uu[0], axis=None)
                skew_uy = stats.skew(mesh.uu[1], axis=None)
                skew_uz = stats.skew(mesh.uu[2], axis=None)
                df_archive=df_archive.append({"rundir":meshdir, "resolution":resolution, "kk":kk,  "relhel":relhel,
                                              "eta":eta, "nu":nu, "time":mesh.timestamp, 
                                              "kurt_ux":kurt_bx, "kurt_uy":kurt_by, "kurt_uz":kurt_bz,
                                              "kurt_bx":kurt_bx, "kurt_by":kurt_by, "kurt_bz":kurt_bz,
                                              "skew_ux":skew_bx, "skew_uy":skew_by, "skew_uz":skew_bz,
                                              "skew_bx":skew_bx, "skew_by":skew_by, "skew_bz":skew_bz}, ignore_index=True)
            else:
                "Error reading file!"


        df_archive.to_csv("tsa_stats%i.csv" % (relhel), index=False) 

        print(df_archive)

