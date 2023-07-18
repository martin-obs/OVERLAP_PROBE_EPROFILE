#!/usr/bin/python
# -*- coding: iso-8859-1 -*-
#
#  Code owner: Martin Osborne
#
#
# (C) Crown Copyright 2022, the Met Office.
#
import matplotlib
#matplotlib.use('agg')
import numpy as np
import datetime
import matplotlib.pyplot as plt
from matplotlib import gridspec
from matplotlib import colors

def create_ceilo_plot ( L1 , vdr = None , mass = None , instrument = None , savepath = None , location = None ) :

    RVal = [255, 212, 209, 207, 205, 202, 199, 196, 193, 189, 186, 183, 179, 176, 172, 169, 166, 163, 159, 156, 153, 149, 146, 143, 139, 136, 132, 128, 124, 121, 117, 113, 109, 105, 102,  98,  93,  89,  85,  81,  77,  73,  70,  66,  61,  57,  53,
            49,  45,  41,  38,  34,  30,  26,  22,  19,  16,  13,  11,   9,   7,   5,   3,   2,   1,   1,   1,   1,   1,   1,   1,   1,   1,   1,   1,   1,   1,   1,   1,   1,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
            0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   1,   1,   1,   1,   1,   1,   1,   1,   0,   0,   0,   0,   0,   0,   0,   0,   0,   2,   4,   7,  10,  14,  18,  23,  28,  34,  41,  48,  55,  63,  72,  80,  88,  95, 103, 112,
            121, 129, 137, 145, 153, 161, 168, 176, 184, 192, 201, 208, 215, 221, 227, 232, 236, 240, 244, 246, 249, 251, 252, 253, 253, 253, 253, 253, 252, 252, 252, 252, 252, 252, 252, 252, 252, 252, 252, 252, 253, 253, 253, 253, 253, 253,
            253, 253, 253, 253, 253, 253, 253, 253, 253, 253, 253, 253, 253, 253, 253, 253, 253, 252, 252, 252, 251, 251, 251, 252, 252, 252, 252, 252, 251, 251, 251, 250, 249, 248, 246, 244, 242, 240, 237, 234, 230, 226, 223, 219, 215, 212,
            208, 204, 199, 195, 191, 188, 184, 180, 176, 172, 167, 163, 159, 155, 151, 147, 144, 141, 138, 136, 134, 132, 131, 130]

    GVal = [255, 216, 212, 209, 206, 203, 200, 197, 194, 190, 187, 184, 180, 176, 173, 169, 166, 163, 159, 156, 153, 149, 146, 142, 139, 135, 132, 128, 124, 121, 117, 113, 110, 106, 102,  98,  94,  90,  86,  82,  78,  74,  70,  67,  62,  58,  54,
            50,  46,  42,  39,  35,  31,  27,  23,  20,  18,  16,  15,  15,  15,  15,  16,  18,  21,  24,  28,  32,  37,  43,  48,  53,  58,  63,  68,  72,  78,  83,  88,  92,  97, 102, 107, 111, 116, 122, 127, 132, 136, 141, 145, 149, 153,
            157, 161, 165, 168, 171, 175, 178, 181, 184, 187, 190, 193, 196, 200, 203, 206, 209, 212, 215, 218, 221, 225, 228, 231, 234, 236, 239, 241, 243, 244, 246, 247, 249, 250, 251, 252, 252, 252, 252, 252, 252, 252, 252, 253, 253, 253,
            253, 253, 253, 253, 253, 253, 253, 253, 253, 253, 253, 253, 253, 252, 251, 249, 247, 245, 243, 241, 238, 235, 232, 229, 225, 221, 217, 213, 209, 205, 201, 196, 192, 188, 184, 180, 176, 172, 168, 164, 161, 156, 152, 148, 144, 141,
            137, 133, 128, 124, 120, 116, 111, 107, 103,  99,  95,  91,  88,  84,  80,  76,  72,  68,  65,  60,  56,  53,  49,  45,  40,  36,  32,  27,  23,  20,  17,  14,  11,   8,   6,   4,   3,   2,   1,   0,   0,   0,   0,   0,   0,   0,
            0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0]

    BVal = [255, 225, 226, 227, 228, 228, 229, 230, 231, 232, 233, 234, 235, 236, 237, 238, 239, 240, 241, 242, 243, 244, 245, 246, 248, 249, 249, 250, 251, 251, 251, 251, 251, 252, 252, 252, 251, 251, 251, 251, 251, 251, 251, 251, 251, 251, 251,
            251, 251, 252, 252, 252, 252, 251, 251, 251, 250, 250, 249, 248, 247, 245, 243, 241, 239, 236, 234, 231, 229, 226, 223, 220, 217, 214, 211, 208, 205, 202, 199, 196, 193, 191, 188, 185, 182, 179, 175, 172, 168, 164, 161, 157, 153,
            149, 145, 141, 137, 133, 128, 123, 118, 113, 108, 102,  97,  92,  86,  81,  76,  71,  66,  61,  56,  51,  46,  41,  36,  32,  28,  24,  20,  17,  14,  12,  10,   8,   6,   4,   3,   2,   2,   1,   1,   1,   1,   1,   1,   1,   1,
            0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   1,   1,   1,   1,   1,   1,   1,   1,   1,   1,   1,   1,   1,   1,   1,   1,   1,   1,   1,   1,   1,   1,   1,   1,   1,   1,   1,   0,   0,   0,   0,   0,   0,
            0,   1,   1,   1,   1,   1,   1,   2,   2,   2,   2,   2,   2,   1,   1,   1,   1,   1,   1,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   1,   1,   2,   3,   5,   7,   9,  11,  13,  16,  19,  22,  26,  30,  33,  36,  40,
            44,  47,  52,  55,  59,  62,  66,  70,  74,  78,  83,  87,  91,  95,  99, 102, 105, 108, 111, 113, 115, 117, 118, 120]

    colours = np.transpose ( np.asarray ( ( RVal , GVal , BVal ) ) )

    fctab= colours / 255.0

    my_cmap = colors.ListedColormap ( fctab , name = 'Cloudnet' , N = None )

    my_cmap.set_bad('white')

    params = {'legend.fontsize': 20,
              'figure.figsize': (15, 5),
              'axes.labelsize': 20,
              'axes.titlesize':20,
              'axes.linewidth':2,
              'xtick.labelsize':30,
              'ytick.labelsize':30,
              'xtick.major.size': 3.5,
              'xtick.minor.size': 2}

    plt.rcParams.update(params)

    date = datetime.datetime.strftime ( L1.dt [ 0 ] , '%Y%m%d' )

    beta = L1.rcs_0

    cbh = L1.cbh

    m_time = L1.Time

    range1 = L1.rng / 1000

    elastic = np.log10(beta)

    Time = np.asarray(m_time)

    if instrument.upper() == 'CL61':

        VDR = np.log10(vdr)

    fig = plt.figure(num=None, facecolor='w', edgecolor='k')

    fig.set_size_inches(15,11)

    LABEL_SIZE = 15

    gs = gridspec.GridSpec(nrows=3, ncols=2 , width_ratios=[1,0.01])

    ax1 = fig.add_subplot(gs[0,0])

    plt.suptitle ( location + ' ' + instrument.upper() + ' ' + date , x = 0.125, y = 0.92,fontsize = LABEL_SIZE, color = 'r', ha = 'left')

    p1 = plt.imshow(np.flipud(np.transpose(elastic)), vmin = 4, vmax = 6, extent=[Time[0],Time[-1],range1[0],range1[-1]],cmap = my_cmap,interpolation='none', aspect = 'auto')

    ax1.xaxis_date()

    cbh_symbols = ['x' , '^' , '*' , 's' , 'o']

    for n in range ( 0, np.shape ( cbh ) [ 1 ] ) :

        ax1.plot ( Time , cbh[:,n]/1000 , cbh_symbols [ n ] , color = 'k', ms = 2 , zorder = 20 )

    date_format = matplotlib.dates.DateFormatter('%H:%M')

    ax1.xaxis.set_major_formatter(date_format)

    plt.title(r'Log$_{10}$ Attenuated Backscatter', fontsize = LABEL_SIZE-4, pad = 10)

    plt.ylabel('Range [km]', fontsize = LABEL_SIZE)

    cax = fig.add_subplot(gs[0,1])

    cbar = matplotlib.colorbar.Colorbar(cax,mappable = p1  , cmap =  my_cmap,orientation='vertical')

    cbar.set_label(r'[m$^{-1}$sr$^{-1}$]', rotation=90, labelpad=20, y=0.45, fontsize = LABEL_SIZE)

    cbar.ax.tick_params(labelsize=15)

    plt.clim(4,6)

    ax1.tick_params(labelsize=LABEL_SIZE-5)

    ax1.set_ylim([0,15])

    if instrument.upper() != 'CL61':

        ax1.set_xlabel('Time [UTC]', fontsize = LABEL_SIZE)

    if instrument.upper() == 'CL31':

        ax1.set_ylim([0,8])

    if instrument.upper() == 'CL61':

        ax2 = plt.subplot(gs[1,0])

        p2 = plt.imshow(np.flipud(np.transpose(VDR)), vmin = -2.5, vmax = 0 , extent=[Time[0],Time[-1],range1[0],range1[-1]],cmap = my_cmap,interpolation='none', aspect = 'auto')

        ax2.xaxis_date()

        date_format = matplotlib.dates.DateFormatter('%H:%M')

        ax2.xaxis.set_major_formatter(date_format)

        plt.title(r'Log$_{10}$VDR', fontsize = LABEL_SIZE-4, pad = 10)

        plt.ylabel('Range [km]', fontsize = LABEL_SIZE)

        cax = fig.add_subplot(gs[1,1])

        cbar = matplotlib.colorbar.Colorbar(cax,mappable = p2  , cmap =  my_cmap,orientation='vertical')

        cbar.set_label(r'[AU]', rotation=90, labelpad=20, y=0.45, fontsize = LABEL_SIZE)

        cbar.ax.tick_params(labelsize=15)

        plt.clim(-2.5,0)

        ax2.tick_params(labelsize=LABEL_SIZE-5)

        ax2.set_ylim([0,15])

        ax3 = plt.subplot(gs[2,0])

        plt.imshow(np.flipud(np.transpose(mass)), extent=[Time[0],Time[-1],range1[0],range1[-1]],cmap = matplotlib.cm.get_cmap('Reds'),interpolation='none', aspect = 'auto')

        ax3.xaxis_date()

        date_format = matplotlib.dates.DateFormatter('%H:%M')

        ax3.xaxis.set_major_formatter(date_format)

        plt.title(r'Mass concentration', fontsize = LABEL_SIZE-4, pad = 10)

        plt.ylabel('Range [km]', fontsize = LABEL_SIZE)

        plt.xlabel('Time [UTC]', fontsize = LABEL_SIZE)

        cax = fig.add_subplot(gs[2,1])

        cbar = matplotlib.colorbar.ColorbarBase(cax,matplotlib.cm.get_cmap('Reds'),orientation='vertical')

        cax.yaxis.set_ticks_position('left')

        tks = [y/200 for y in range ( 0 , 200 ,25) ]

        cax.set_yticks(tks)

        fac = 0.68/0.38

        tk_lab = [ int(y*fac) for y in range ( 0 , 240 ,40) ]

        cax.set_yticklabels(tk_lab)

        cax.set_ylabel(r'Ash [$\mu$gm$^{-3}$]', rotation=90, labelpad=30, y=0.45, fontsize = LABEL_SIZE-5)

        clone = cax.twinx()

        clone.tick_params(axis='both', which='major', labelsize=10)

        clone.set_ylim([0,200])

        clone.set_ylabel(r'Dust [$\mu$gm$^{-3}$]', rotation=90, labelpad=-70, y=0.45, fontsize = LABEL_SIZE-5)

        cbar.ax.tick_params(labelsize=LABEL_SIZE-5)

        plt.clim(0,200)

        ax3.tick_params(labelsize=LABEL_SIZE-5)

        ax3.set_ylim([0,15])

    fig.subplots_adjust(wspace=0.11)

    fig.savefig ( savepath + '/' + instrument.upper ( ) + '_' + date + '.png' , bbox_inches = 'tight' , format = 'png' , dpi = 300 )
