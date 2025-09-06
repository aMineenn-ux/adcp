#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar 26 23:39:23 2023

@author: sendy
"""

import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from pathlib import Path
import re
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import matplotlib.pylab as pylab
params = {
   'axes.labelsize': 8,
   'font.size': 10,
   'legend.fontsize': 10,
   'xtick.labelsize': 10,
   'ytick.labelsize': 10,
   'text.usetex': False,
   'figure.figsize': [7, 3]
   }
pylab.rcParams.update(params)

def deldir(df):
    # select direction angle between data more than 30d and replace with nan
    # Find the differences between adjacent values
    diff = df['dir'].diff()

    # Handle the wrap-around case using vectorization
    diff = np.where(diff > 180, diff - 360, diff)
    diff = np.where(diff < -180, diff + 360, diff)

    # Replace the differences greater than 45 with NaN
    df['dir'].loc[np.absolute(diff) > 45] = np.nan
    df.dropna(inplace=True)
    return df

def clean_dir(df):
    # apply deldir to every ensamble
    # use dataframe from process_line function
    df = df.reset_index(drop=True)
    clean_dir = df.groupby('ens').apply(deldir).reset_index(drop=True)
    clean_dir = clean_dir.drop(['Height', 'hn', 'hbn'], axis=1)
    return clean_dir

def proc_shear(df):
    # calculate shear velocity and z0
    # use dataframe df from process_line function
    z = np.array(df['z']).reshape((-1,1))
    ln_z = np.log(z)
    uz = df.vel
    model = LinearRegression().fit(ln_z, uz)
    us = np.round(0.4*model.coef_,3).tolist()
    z0 = np.round(np.exp(-model.intercept_/model.coef_),3).tolist()
    return us, z0

def shear(df, file_name):
    outdav = Path('Orde_1 - CP_PRODUCT/Properties/Shear_friction')
    outdav.mkdir(parents=True, exist_ok=True)
        
    # apply to every ensamble
    dum = list()
    df = df.reset_index(drop=True)
    dfl = df.groupby(['ens'])
    for ens, ens_df in dfl:
        us, z0 = proc_shear(ens_df)
        ens_df.loc[:,'us'] = us *len(ens_df)
        ens_df.loc[:,'z0'] = z0 *len(ens_df)
        dum.append(ens_df)
    shear = pd.concat(dum, ignore_index=True)
    
    shear_raw_out = re.sub(r'(?i).txt', '_raw_shear.csv', file_name)
    shear.to_csv(outdav / shear_raw_out, index=False)
    print(f'* Finished save raw Shear Friction file {shear_raw_out}')
    return shear

def near_seabed(df, direction):
    # select data up to seabed after clean up difference direction criteria
    # select data with u* and z0 more than 0
    dum = list()
    dfl = df.groupby(['ens'])
    for ens, ens_df in dfl:
        # select shear friction (u*) and z0 > 0
        ens_df.loc[(ens_df['us'] < 0) | (ens_df['z0'] < 0) | (ens_df['z0'] > ens_df['h'])] = np.nan
        
        # select data with criteria bin height above 12% side lobe interference
        if direction == 'normal':
            real_lastbin = ens_df['hb'].iloc[-1]
            cal_lastbin = (ens_df['h'].iloc[-1]) - (ens_df['h'].iloc[-1]*(12/100))  #  used for normal direction
        else:
            real_lastbin = ens_df['hb'].iloc[0]
            cal_lastbin = (ens_df['h'].iloc[0]) - (ens_df['h'].iloc[0]*(12/100))  #  used for reversed direction

        if real_lastbin > cal_lastbin:
            dum.append(ens_df)
        else:
            continue
            
    # data for turbidity close to seabed
    df_sbd = pd.concat(dum, ignore_index=True) 
    df_sbd = df_sbd.dropna()
    return df_sbd

def turbulence(df, dir_ln, file_name):
    # select data near seabed from df data
    #df_nsbd = near_seabed(df, dir_ln)
    
    #process calculation turbidity
    var = df.groupby('ens')[['u', 'v', 'w']].transform(np.var)
    U = np.sqrt(df['u']**2 + df['v']**2 + df['w']**2)
    df['turb'] = ((np.sqrt(1/3 * var.sum(axis=1)))/U)*100
    
    outdav = Path('Orde_1 - CP_PRODUCT/Properties/Turbulence')
    outdav.mkdir(parents=True, exist_ok=True)
    turb_out = re.sub(r'(?i).txt', '_Turbulence.csv', file_name)
    
    df.to_csv(outdav / turb_out, index=False)
    print(f'* Finished save Turbulence file {turb_out}')
    return df

def turb_plot(df, gdf, file_name, lbl_start, lbl_end):
    arr_col = df.pivot_table(index='hb', columns='dist', values='turb')
    array = np.array(arr_col)
    xax = np.array(arr_col.columns)
    yax = np.array(arr_col.index)
    
    outplot = Path('Orde_1 - CP_PRODUCT/Properties/Turbulence/plot')
    outplot.mkdir(parents=True, exist_ok=True)
    turb_out = re.sub(r'(?i).txt', '_turb.jpg', file_name)
    linename = Path(file_name).stem
    
    fig, ax = plt.subplots()
    fig.suptitle('Turbulence Intensity - ' + linename)
    img = ax.pcolormesh(xax, yax, array, cmap='rainbow', vmin=0, vmax=100)
    ax.invert_yaxis()
    
    # plot MSL
    ax.axhline(y=0, color='k', lw=1, linestyle='--')
    ax.plot(gdf.dist_tot, gdf.Height*-1,'r')

    # plot seabed profile 
    zmax = gdf.hn.max() +5
    ax.plot(gdf.dist_tot, gdf.hn,'k')
    ax.fill_between(gdf.dist_tot, gdf.hn, zmax, color='slategrey')
    
    # setting label
    bbox = dict(facecolor='none', edgecolor='black')
    ax.annotate(lbl_start, xy=(0, 0), xycoords='axes fraction',xytext=(-0.01, -0.25), bbox = bbox)
    ax.annotate(lbl_end, xy=(0, 0), xycoords='axes fraction',xytext=(0.99, -0.25),bbox = bbox)
    ax.set_ylim(zmax, -5.0)
    ax.set_xlabel('Distance (m)')
    ax.set_ylabel('Depth (m)')

    # define colorbar properties
    axins = inset_axes(ax, width="80%", height="5%",loc='lower center', borderpad=-5)

    # plot colorbar
    fig.colorbar(img, cax=axins, orientation="horizontal", label='Turbulence Intensity (%)')

    fig.savefig(outplot / turb_out, bbox_inches='tight', dpi=150)
    plt.close()
    return print(f"-> {turb_out} image saved")