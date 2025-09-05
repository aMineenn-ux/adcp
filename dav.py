#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr  7 22:50:55 2023

@author: sendy
"""
import pandas as pd
import numpy as np
import geopandas as gpd
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from pathlib import Path
import re
import matplotlib.pylab as pylab

params = {
   'axes.labelsize': 8,
   'font.size': 10,
   'legend.fontsize': 10,
   'xtick.labelsize': 10,
   'ytick.labelsize': 10,
   'text.usetex': False,
   'figure.figsize': [12, 3]
   }
pylab.rcParams.update(params)

def compute_dav(ens_df, h):
    # DAV calculation
    ens_df['dav'] = (((1/h)*(np.trapz(ens_df['vel'], x=ens_df['z'])))*-1)  #  direction as it is
    ens_df['avg_u'] = ens_df['u'].mean()
    ens_df['avg_v'] = ens_df['v'].mean()
    ens_df['avg_w'] = ens_df['w'].mean() 
    return ens_df

def compute_dav_brg(df):
    # compute dav angle direction
    df['bearing'] = np.degrees(np.arctan(df['avg_u']/df['avg_v']))
    df.loc[(df['avg_u']>0) & (df['avg_v']>0), 'mdir'] = df['bearing']
    df.loc[((df['avg_u']>0) & (df['avg_v']<0)) | ((df['avg_u']<0) & (df['avg_v']<0)), 'mdir'] = (df['bearing']+180) %360
    df.loc[(df['avg_u']<0) & (df['avg_v']>0), 'mdir'] = (df['bearing']+360) %360
    df = df.drop(['bearing'], axis=1)
    return df

def proc_dav(df, file_name):
    # process DAV and export to file
    outdav= Path('Orde_1 - CP_PRODUCT/DAV')
    outdav.mkdir(parents=True, exist_ok=True)
    dav_out = re.sub(r'(?i).txt', '_dav.csv', file_name)
        
    df = df.reset_index(drop=True)
    line = df.groupby(by=['ens']).apply(lambda x: compute_dav(x, x['h'].max())).reset_index(drop=True)
    dav_dat = line.drop_duplicates(subset='ens')
    dav_dat = compute_dav_brg(dav_dat)
    dav_dat = dav_dat[['date', 'ens', 'dist', 'lat', 'lon', 'avg_u', 'avg_v', 'mdir', 'dav', 'us', 'z0']]
    dav_dat.to_csv(outdav / dav_out, index=False)
    print(f'* Finished save DAV file {dav_out}')
    return dav_dat

def dav_to_shp(dat, roll_num, file_name):
    outshp= Path('Orde_1 - CP_PRODUCT/DAV/dav_shp')
    outshp.mkdir(parents=True, exist_ok=True)
    davshp_out = re.sub(r'(?i).txt', '_dav.shp', file_name)
    
    dat.drop('date', axis=1, inplace=True)
    dat[['dav_roll', 'mdir_roll']] = dat[['dav', 'mdir']].rolling(roll_num).mean()
    dat['dav_roll'].fillna(dat['dav'], inplace=True)
    dat['mdir_roll'].fillna(dat['mdir'], inplace=True)
    df = dat.loc[::roll_num].reset_index(drop=True)
    
    gdf = gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(df.lon, df.lat), crs='EPSG:4326')
    pt_gdf = gdf[['geometry','ens', 'dist', 'dav_roll', 'mdir_roll']].copy()
    
    pt_gdf.to_file(outshp / davshp_out, encoding='utf-8')
    return print(f'* Finished save dav to shapefile {davshp_out}')

def davplot(dav, file_name):
    outplot= Path('Orde_1 - CP_PRODUCT/DAV/plot')
    outplot.mkdir(parents=True, exist_ok=True)
    davplot_out = re.sub(r'(?i).txt', '_feather.jpg', file_name)
    dav_roll_out = re.sub(r'(?i).txt', '_dav20.csv', file_name)
    dav = dav.reset_index(drop=True)
    dav['zero'] = pd.Series([0 for x in range(len(dav.index))])

    dav['dav_20'] = dav['dav'].rolling(20).mean()
    # line['mdir_20'] = line['mdir'].rolling(20).mean()
    dav.loc[0, 'dav_20'] = dav['dav'].loc[0]
    # line.loc[0, 'mdir_20'] = line['mdir'].loc[0]

    df = dav.iloc[::20] #  set every n ensemble
    
    df.to_csv(outplot / dav_roll_out, index=False)
    
    linename = Path(file_name).stem
    print(f'-> Process plot DAV {linename}, Please Wait!')
    fig, ax = plt.subplots()
    fig.suptitle(f'Feather Plot {str(linename)} (20 Ensemble - Average)')
    dav = df['dav_20']
    u = df["dav_20"] * np.sin(np.radians(df["mdir"]))
    v = df["dav_20"] * np.cos(np.radians(df["mdir"]))

    norm = colors.Normalize(vmin=0.1, vmax=2.0)
    Q = ax.quiver(df['dist'], df['zero'], u, v, dav, width=0.002, cmap='jet', norm=norm, scale=1 / 0.1)
    ax.quiverkey(Q, 0.2, 0.91, 0.5, r'East 0.5 m/s', labelpos='E',coordinates='figure')
    ax.plot(df['dist'], df['zero'], 'k', linewidth=0.5)
    ax.set_xlabel('Distance (m)')
    ax.yaxis.set_ticks([])
    ax.tick_params(axis="x", direction='in', length=5)
    ax.set_xlim(-1000, 12500)
    ax.set_ylim(v.min()-1.5, v.max()+1.5)
    # plt.ylim(-2, 2)

    # define colorbar properties
    axins = inset_axes(ax, width="80%", height="5%", loc='lower center', borderpad=-5)

    # plot colorbar
    fig.colorbar(Q, cax=axins, orientation="horizontal", label='Depth-average Velocity m/s')

    fig.savefig(outplot / davplot_out, bbox_inches='tight', dpi=150)
    plt.close()
    return print(f'-> Plotting DAV {linename}, DONE!')