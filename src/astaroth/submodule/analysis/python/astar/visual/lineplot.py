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

end_rm = -1 #-35#-40

def plot_min_man_rms(ts, xaxis, yaxis1, yaxis2, yaxis3):
    if yaxis1 in ts.var:
        plt.plot(ts.var[xaxis][:end_rm], ts.var[yaxis1][:end_rm], label=yaxis1)
        plt.plot(ts.var[xaxis][:end_rm], ts.var[yaxis2][:end_rm], label=yaxis2)
        plt.plot(ts.var[xaxis][:end_rm], ts.var[yaxis3][:end_rm], label=yaxis3)
        plt.xlabel(xaxis)
        plt.legend()
    else:
        print("%s %s and %s not found! Skipping...", yaxis1, yaxis2, yaxis3)
        plt.close() 

def plot_ts(ts, isotherm=False, show_all=False, lnrho=False, uutot=False, 
            uux=False, uuy=False, uuz=False, 
            aax=False, aay=False, aaz=False, 
            ss=False, acc=False, sink=False, rho=False, bb=False, alfven=False):

    if show_all:
        lnrho=True
        rho=True
        uutot=True
        uux=True
        uuy=True
        uuz=True
        ss=True
        acc=True
        sink=True
        bb=True
        alfven=True
        shock=True

    if isotherm:
        lnrho=True
        rho=True
        uutot=True
        uux=True
        uuy=True
        uuz=True
        ss=False
        acc=True
        sink=True

    if lnrho:
        plt.figure()
        xaxis  = 't_step'
        yaxis1 = 'lnrho_rms'
        yaxis2 = 'lnrho_min'
        yaxis3 = 'lnrho_max'
        plot_min_man_rms(ts, xaxis, yaxis1, yaxis2, yaxis3)

    if rho:
        plt.figure()
        xaxis  = 't_step'
        yaxis1 = 'rho_rms'
        yaxis2 = 'rho_min'
        yaxis3 = 'rho_max'
        plot_min_man_rms(ts, xaxis, yaxis1, yaxis2, yaxis3)
     
    if uutot:   
        plt.figure()
        xaxis = 't_step'
        yaxis1 = 'uutot_rms'
        yaxis2 = 'uutot_min'
        yaxis3 = 'uutot_max'
        plot_min_man_rms(ts, xaxis, yaxis1, yaxis2, yaxis3)
        
    if uux:   
        plt.figure()
        xaxis = 't_step'
        yaxis1 = 'uux_rms'
        yaxis2 = 'uux_min'
        yaxis3 = 'uux_max'
        plot_min_man_rms(ts, xaxis, yaxis1, yaxis2, yaxis3)
        
    if uuy:   
        plt.figure()
        xaxis = 't_step'
        yaxis1 = 'uuy_rms'
        yaxis2 = 'uuy_min'
        yaxis3 = 'uuy_max'
        plot_min_man_rms(ts, xaxis, yaxis1, yaxis2, yaxis3)
        
    if uuz:   
        plt.figure()
        xaxis = 't_step'
        yaxis1 = 'uuz_rms'
        yaxis2 = 'uuz_min'
        yaxis3 = 'uuz_max'
        plot_min_man_rms(ts, xaxis, yaxis1, yaxis2, yaxis3)

    if aax:   
        plt.figure()
        xaxis = 't_step'
        yaxis1 = 'aax_rms'
        yaxis2 = 'aax_min'
        yaxis3 = 'aax_max'
        plot_min_man_rms(ts, xaxis, yaxis1, yaxis2, yaxis3)
        
    if aay:   
        plt.figure()
        xaxis = 't_step'
        yaxis1 = 'aay_rms'
        yaxis2 = 'aay_min'
        yaxis3 = 'aay_max'
        plot_min_man_rms(ts, xaxis, yaxis1, yaxis2, yaxis3)
        
    if aaz:   
        plt.figure()
        xaxis = 't_step'
        yaxis1 = 'aaz_rms'
        yaxis2 = 'aaz_min'
        yaxis3 = 'aaz_max'
        plot_min_man_rms(ts, xaxis, yaxis1, yaxis2, yaxis3)
  
    if ss:   
        plt.figure()
        xaxis = 't_step'
        yaxis1 = 'ss_rms'
        yaxis2 = 'ss_min'
        yaxis3 = 'ss_max'
        plot_min_man_rms(ts, xaxis, yaxis1, yaxis2, yaxis3)

    if bb:   
        plt.figure()
        xaxis = 't_step'
        yaxis1 = 'bbtot_rms'
        yaxis2 = 'bbtot_min'
        yaxis3 = 'bbtot_max'
        plot_min_man_rms(ts, xaxis, yaxis1, yaxis2, yaxis3)

        plt.figure()
        xaxis = 't_step'
        yaxis1 = 'bbx_rms'
        yaxis2 = 'bbx_min'
        yaxis3 = 'bbx_max'
        plot_min_man_rms(ts, xaxis, yaxis1, yaxis2, yaxis3)

        plt.figure()
        xaxis = 't_step'
        yaxis1 = 'bby_rms'
        yaxis2 = 'bby_min'
        yaxis3 = 'bby_max'
        plot_min_man_rms(ts, xaxis, yaxis1, yaxis2, yaxis3)

        plt.figure()
        xaxis = 't_step'
        yaxis1 = 'bbz_rms'
        yaxis2 = 'bbz_min'
        yaxis3 = 'bbz_max'
        plot_min_man_rms(ts, xaxis, yaxis1, yaxis2, yaxis3)

    if alfven:
        plt.figure()
        xaxis = 't_step'
        yaxis1 = 'vAtot_rms'
        yaxis2 = 'vAtot_min'
        yaxis3 = 'vAtot_max'
        plot_min_man_rms(ts, xaxis, yaxis1, yaxis2, yaxis3)


    if acc:   
        plt.figure()
        xaxis = 't_step'
        yaxis1 = 'acc_rms'
        yaxis2 = 'acc_min'
        yaxis3 = 'acc_max'
        plot_min_man_rms(ts, xaxis, yaxis1, yaxis2, yaxis3)

        plt.figure()
        xaxis = 't_step'
        yaxis1 = 'sink_mass'
        yaxis2 = 'sink_mass'
        yaxis3 = 'sink_mass'
        plot_min_man_rms(ts, xaxis, yaxis1, yaxis2, yaxis3)
  
        plt.figure()
        xaxis = 't_step'
        yaxis1 = 'accreted_mass'
        yaxis2 = 'accreted_mass'
        yaxis3 = 'accreted_mass'
        plot_min_man_rms(ts, xaxis, yaxis1, yaxis2, yaxis3)
  
    if shock:
        plt.figure()
        xaxis = 't_step'
        yaxis1 = 'shock_rms'
        yaxis2 = 'shock_min'
        yaxis3 = 'shock_max'
        plot_min_man_rms(ts, xaxis, yaxis1, yaxis2, yaxis3)



    plt.show()
