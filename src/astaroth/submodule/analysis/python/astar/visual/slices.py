
'''
    Copyright (C) 2014-2021, Johannes Pekkila, Miikka Vaisala.

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
import pylab as plt 
import numpy as np 
import matplotlib.gridspec as gridspec
import matplotlib.colors as colors

CM_INFERNO = plt.get_cmap('inferno')

def plot_3(mesh, input_grid, title = '', fname = 'default', bitmap=False,
	   slicetype = 'middle', colrange=None, colormap=CM_INFERNO ,
           contourplot=False, points_from_centre = -1, bfieldlines=False, velfieldlines=False, trimghost = 0, trimaxis = 0 ):

    fig = plt.figure(figsize=(8, 8))
    grid = gridspec.GridSpec(2, 3, wspace=0.4, hspace=0.4, width_ratios=[1,1, 0.15])
    ax00   = fig.add_subplot( grid[0,0] )
    ax10   = fig.add_subplot( grid[0,1] )
    ax11   = fig.add_subplot( grid[1,1] )
    axcbar = fig.add_subplot( grid[:,2] )

    print(mesh.minfo.contents.keys())

    if slicetype == 'middle':
        yz_slice = input_grid[mesh.xmid, :, :]
        xz_slice = input_grid[:, mesh.ymid, :]
        xy_slice = input_grid[:, :, mesh.zmid]
    elif slicetype == 'sum':
        yz_slice = np.sum(input_grid, axis=0) 
        xz_slice = np.sum(input_grid, axis=1) 
        xy_slice = np.sum(input_grid, axis=2) 

    yz_slice = yz_slice[trimghost : yz_slice.shape[0]-trimghost,
                        trimghost : yz_slice.shape[1]-trimghost]
    xz_slice = xz_slice[trimghost : xz_slice.shape[0]-trimghost,
                        trimghost : xz_slice.shape[1]-trimghost]
    xy_slice = xy_slice[trimghost : xy_slice.shape[0]-trimghost,
                        trimghost : xy_slice.shape[1]-trimghost]
    mesh_xx_tmp  = mesh.xx[(trimghost + trimaxis) : mesh.xx.shape[0]-(trimghost + trimaxis)] 
    mesh_yy_tmp  = mesh.yy[(trimghost + trimaxis) : mesh.yy.shape[0]-(trimghost + trimaxis)]  
    mesh_zz_tmp  = mesh.zz[(trimghost + trimaxis) : mesh.zz.shape[0]-(trimghost + trimaxis)] 

    #Set the coulourscale
    cmin = np.amin([yz_slice.min(), xz_slice.min(), xy_slice.min()])
    cmax = np.amax([yz_slice.max(), xz_slice.max(), xy_slice.max()])
    if colrange==None:
        plotnorm = colors.Normalize(vmin=cmin,vmax=cmax) 
    else:
        plotnorm = colors.Normalize(vmin=colrange[0],vmax=colrange[1]) 

    
    if points_from_centre > 0:
        yz_slice = yz_slice[int(yz_slice.shape[0]/2)-points_from_centre : int(yz_slice.shape[0]/2)+points_from_centre,
                            int(yz_slice.shape[1]/2)-points_from_centre : int(yz_slice.shape[1]/2)+points_from_centre]
        xz_slice = xz_slice[int(xz_slice.shape[0]/2)-points_from_centre : int(xz_slice.shape[0]/2)+points_from_centre,
                            int(xz_slice.shape[1]/2)-points_from_centre : int(xz_slice.shape[1]/2)+points_from_centre]
        xy_slice = xy_slice[int(xy_slice.shape[0]/2)-points_from_centre : int(xy_slice.shape[0]/2)+points_from_centre,
                            int(xy_slice.shape[1]/2)-points_from_centre : int(xy_slice.shape[1]/2)+points_from_centre]
        mesh_xx_tmp = mesh.xx[int(mesh.xx.shape[0]/2)-points_from_centre : int(mesh.xx.shape[0]/2)+points_from_centre] 
        mesh_yy_tmp = mesh.yy[int(mesh.yy.shape[0]/2)-points_from_centre : int(mesh.yy.shape[0]/2)+points_from_centre]  
        mesh_zz_tmp = mesh.zz[int(mesh.zz.shape[0]/2)-points_from_centre : int(mesh.zz.shape[0]/2)+points_from_centre] 
    
    yy, zz = np.meshgrid(mesh_xx_tmp, mesh_xx_tmp, indexing='ij')
    if contourplot:
        map1 = ax00.contourf(yy, zz, yz_slice, norm=plotnorm, cmap=colormap, nlev=10)
    else:
        map1 = ax00.pcolormesh(yy, zz, yz_slice, norm=plotnorm, cmap=colormap)
    ax00.set_xlabel('y')
    ax00.set_ylabel('z')
    ax00.set_title('%s t = %.4e' % (title, mesh.timestamp) )    
    ax00.set_aspect('equal')

    if mesh.minfo.contents["AC_accretion_range"] > 0.0:
        ax00.contour(yy, zz, np.sqrt((yy-yy.max()/2.0)**2.0 + (zz-zz.max()/2.0)**2.0), [mesh.minfo.contents["AC_accretion_range"]]) 
    
    xx, zz = np.meshgrid(mesh_xx_tmp, mesh_zz_tmp, indexing='ij')
    if contourplot:
        ax10.contourf(xx, zz, xz_slice, norm=plotnorm, cmap=colormap, nlev=10)
    else:
        ax10.pcolormesh(xx, zz, xz_slice, norm=plotnorm, cmap=colormap)
    ax10.set_xlabel('x')
    ax10.set_ylabel('z')
    ax10.set_aspect('equal')

    if mesh.minfo.contents["AC_accretion_range"] > 0.0:
        ax10.contour(xx, zz, np.sqrt((xx-xx.max()/2.0)**2.0 + (zz-zz.max()/2.0)**2.0), [mesh.minfo.contents["AC_accretion_range"]]) 
    
    xx, yy = np.meshgrid(mesh_xx_tmp, mesh_yy_tmp, indexing='ij')
    if contourplot:
        ax11.contourf(xx, yy, xy_slice, norm=plotnorm, cmap=colormap, nlev=10)
    else:
        ax11.pcolormesh(xx, yy, xy_slice, norm=plotnorm, cmap=colormap)
    ax11.set_xlabel('x')
    ax11.set_ylabel('y')
    ax11.set_aspect('equal')

    if mesh.minfo.contents["AC_accretion_range"] > 0.0:
        ax11.contour(xx, yy, np.sqrt((xx-xx.max()/2.0)**2.0 + (yy-yy.max()/2.0)**2.0), [mesh.minfo.contents["AC_accretion_range"]]) 

    if bfieldlines:
        ax00.streamplot(mesh.yy, mesh.zz, np.mean(mesh.bb[1], axis=0), np.mean(mesh.bb[2], axis=0))
        ax10.streamplot(mesh.xx, mesh.zz, np.mean(mesh.bb[0], axis=1), np.mean(mesh.bb[2], axis=1))
        ax11.streamplot(mesh.xx, mesh.yy, np.mean(mesh.bb[0], axis=2), np.mean(mesh.bb[1], axis=2))

        #ax00.streamplot(mesh.yy, mesh.zz, mesh.bb[1][mesh.xmid, :, :], mesh.bb[2][mesh.xmid, :, :])
        #ax10.streamplot(mesh.xx, mesh.zz, mesh.bb[0][:, mesh.ymid, :], mesh.bb[2][:, mesh.ymid, :])
        #ax11.streamplot(mesh.xx, mesh.yy, mesh.bb[0][:, : ,mesh.zmid], mesh.bb[1][:, :, mesh.zmid])

        #ax00.quiver(mesh.bb[2][mesh.xmid, ::10, ::10], mesh.bb[1][mesh.xmid, ::10, ::10], pivot='middle')
        #ax10.quiver(mesh.bb[2][::10, mesh.ymid, ::10], mesh.bb[0][::10, mesh.ymid, ::10], pivot='middle')
        #ax11.quiver(mesh.bb[1][::10, ::10, mesh.zmid], mesh.bb[0][::10, ::10, mesh.zmid], pivot='middle')

        #ax00.quiver(mesh.yy, mesh.zz, mesh.bb[2][mesh.xmid, :, :], mesh.bb[1][mesh.xmid, :, :], pivot='middle')
        #ax10.quiver(mesh.xx, mesh.zz, mesh.bb[2][:, mesh.ymid, :], mesh.bb[0][:, mesh.ymid, :], pivot='middle')
        #ax11.quiver(mesh.xx, mesh.yy, mesh.bb[1][:, :, mesh.zmid], mesh.bb[0][:, :, mesh.zmid], pivot='middle')

    if velfieldlines:
        ax00.streamplot(mesh.yy, mesh.zz, mesh.uu[2][mesh.xmid, :, :], mesh.uu[1][mesh.xmid, :, :])
        ax10.streamplot(mesh.xx, mesh.zz, mesh.uu[2][:, mesh.ymid, :], mesh.uu[0][:, mesh.ymid, :])
        ax11.streamplot(mesh.xx, mesh.yy, mesh.uu[1][:, :, mesh.zmid], mesh.uu[0][:, : ,mesh.zmid])
    
    cbar = plt.colorbar(map1, cax=axcbar) 

    if bitmap:
        plt.savefig('%s_%s.png' % (fname, mesh.framenum))
        print('Saved %s_%s.png' % (fname, mesh.framenum))
        plt.close(fig)
         
def volume_render(mesh, val1 = {"variable": None, "min": None, "max":None, "opacity":1.0}):
        
    if val1["variable"] == "btot":
        plt.figure()
        bb_tot = np.sqrt(mesh.bb[0]**2.0 + mesh.bb[1]**2.0 + mesh.bb[2]**2.0)
        array = bb_tot
        varname = "btot"
        meshxx = mesh.xx[3:-3]
        meshyy = mesh.yy[3:-3]
        meshzz = mesh.zz[3:-3]

    if val1["variable"] == "utot":
        plt.figure()
        uu_tot = np.sqrt(mesh.uu[0]**2.0 + mesh.uu[1]**2.0 + mesh.uu[2]**2.0)
        array = uu_tot
        varname = "utot"
        meshxx = mesh.xx
        meshyy = mesh.yy
        meshzz = mesh.zz

    if val1["variable"] == "rho":
        plt.figure()
        array = np.exp(mesh.lnrho)
        varname = "rho"
        meshxx = mesh.xx
        meshyy = mesh.yy
        meshzz = mesh.zz

    if val1["variable"] == "aa":
        plt.figure()
        aa_tot = np.sqrt(mesh.aa[0]**2.0 + mesh.aa[1]**2.0 + mesh.aa[2]**2.0)
        array = aa_tot
        varname = "aa"
        meshxx = mesh.xx
        meshyy = mesh.yy
        meshzz = mesh.zz

    #Histogram plot to find value ranges. 
    hist, bedges = np.histogram(array, bins=mesh.xx.size)
    plt.plot(bedges[:-1], hist)
    plt.yscale('log')
    if val1["min"] != None or val1["max"] != None:
        plt.plot([val1["min"],val1["min"]], [1,hist.max()], label=varname+" min")
        plt.plot([val1["max"],val1["max"]], [1,hist.max()], label=varname+" max")
        plt.legend()

    plt.savefig('volrend_hist_%s_%s.png' % (varname, mesh.framenum))
    plt.close()

    if val1["min"] != None or val1["max"] != None:

        #print(np.where(bb_tot < val1["min"]))
    
        array[np.where(array < val1["min"])] = 0.0
        array[np.where(array > val1["max"])] = 0.0
        array[np.where(array > 0.0)] = val1["opacity"]
    
        #plt.figure()
        #plt.plot(bb_tot[:,64,64])

        mapyz = array.sum(axis=0)
        mapxz = array.sum(axis=1)
        mapxy = array.sum(axis=2)

        yy_yz, zz_yz = np.meshgrid(meshyy, meshzz, indexing='ij')
        xx_xz, zz_xz = np.meshgrid(meshxx, meshzz, indexing='ij')
        xx_xy, yy_xy = np.meshgrid(meshxx, meshyy, indexing='ij')

        fig, ax = plt.subplots()
        #plt.imshow(mapyz, vmin=0.0, vmax=1.0)
        plt.pcolormesh(yy_yz, zz_yz, mapyz, vmin=0.0, vmax=1.0, shading='auto')
        ax.set_aspect('equal')
        ax.set_title(varname)
        ax.set_xlabel('y')
        ax.set_ylabel('z')
        plt.savefig('volrend_%s_%s_%s.png' % (varname, "yz", mesh.framenum))
        plt.close()
    
        fig, ax = plt.subplots()
        #plt.imshow(mapxz, vmin=0.0, vmax=1.0)
        plt.pcolormesh(xx_xz, zz_xz, mapxz, vmin=0.0, vmax=1.0, shading='auto')
        ax.set_aspect('equal')
        ax.set_title(varname)
        ax.set_xlabel('x')
        ax.set_ylabel('z')
        plt.savefig('volrend_%s_%s_%s.png' % (varname, "xz", mesh.framenum))
        plt.close()
    
        fig, ax = plt.subplots()
        #plt.imshow(mapxy, vmin=0.0, vmax=1.0)
        plt.pcolormesh(xx_xy, yy_xy, mapxy, vmin=0.0, vmax=1.0, shading='auto')
        ax.set_aspect('equal')
        ax.set_title(varname)
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        plt.savefig('volrend_%s_%s_%s.png' % (varname, "xy", mesh.framenum))
        plt.close()

    #plt.show()
 
