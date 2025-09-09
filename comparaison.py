#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Feb  17 22:11:18 2023

@author: sendy

This script is use for process transect file from reformat result,
process the tide data for plotting, calculate sea current DAV,
save the ploting image, save DAV calulating result and save transect line 
to ESRI shapefile
"""

import pandas as pd
import geopandas as gpd
import numpy as np
from shapely.geometry import LineString
from math import degrees, atan
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import glob
from pathlib import Path
import re
import dav
import currentprop
import avg_5m
import warnings
import matplotlib.pylab as pylab
import os
OUTPATH = Path(os.environ.get("ADCP_OUTPATH", "Orde_1 - CP_PRODUCT"))
OUTPATH.mkdir(parents=True, exist_ok=True)

params = {
   'axes.labelsize': 8,
   'font.size': 10,
   'legend.fontsize': 10,
   'xtick.labelsize': 10,
   'ytick.labelsize': 10,
   'text.usetex': False,
   'figure.figsize': [12, 2.5]
   }
pylab.rcParams.update(params)

#suppress warnings
warnings.filterwarnings('ignore')

# REMPLACEZ l'ancienne fonction process_tide par celle-ci
def process_tide(line, tide):
    # Get t_sol (start of line) and t_eol (end of line)
    t_sol = line.date.min()
    t_eol = line.date.max()

    print(f"    -> Période du transect ADCP : de {t_sol} à {t_eol}")

    # parse the date from the 'Time' column and set as index
    tide['Time'] = pd.to_datetime(tide['Time'], format='%Y-%m-%d %H:%M:%S')
    tide.set_index('Time', inplace=True)
    
    print(f"    -> Période du fichier de marée : de {tide.index.min()} à {tide.index.max()}")

    # resample to every second and interpolate
    tide_res = tide.resample('1S').asfreq()
    tidenew = tide_res.interpolate().round(3)
        
    # masking tide data between time of line transect
    mask = (tidenew.index >= t_sol) & (tidenew.index <= t_eol)
    masked_tl = tidenew.loc[mask]
    masked_tl.index.name = 'date'

    # --- CORRECTION DE ROBUSTESSE ICI ---
    # On vérifie si la DataFrame masquée est vide
    if masked_tl.empty:
        print("    -> AVERTISSEMENT : Aucune donnée de marée trouvée pour la période du transect.")
        print("    -> La correction de marée sera nulle (hauteur = 0).")
        # On crée une DataFrame vide avec les bonnes colonnes pour que le reste du script fonctionne
        # On utilise l'index de 'line' pour que la fusion (merge_asof) fonctionne
        masked_tl = pd.DataFrame(index=line['date'])
        masked_tl['Height'] = 0.0
        xc = t_sol  # Valeur factice
        yc = 0.0     # Valeur factice
    else:
        print(f"    -> {len(masked_tl)} points de données de marée trouvés pour la période du transect.")
        # locate center of mask data
        xc = masked_tl.index[int(len(masked_tl)/2)]
        yc = masked_tl['Height'].iloc[len(masked_tl) // 2]
    
    return t_sol, t_eol, masked_tl, tidenew, xc, yc

def dir_vel_zdist(ens_df):
    ens_df['z'] = ens_df['h'] - ens_df['hb'] # calculate height above seabed
    ens_df['z'] = ens_df['z'].replace(ens_df['z'].max(), ens_df['h'].max()) # replace max height with depth
    ens_df['z'] = round(ens_df['z'],3)

    # calculate velocity and direction from vector
    ens_df['vel'] = np.sqrt((ens_df['u']**2)+(ens_df['v']**2)).round(3)
    ens_df['bearing'] = np.degrees(np.arctan(ens_df['u']/ens_df['v'])).round(3)
    ens_df.loc[(ens_df['u']>0) & (ens_df['v']>0), 'dir'] = ens_df['bearing']
    ens_df.loc[((ens_df['u']>0) & (ens_df['v']<0)) | ((ens_df['u']<0) & (ens_df['v']<0)), 'dir'] = (ens_df['bearing']+180) %360
    ens_df.loc[(ens_df['u']<0) & (ens_df['v']>0), 'dir'] = (ens_df['bearing']+360) %360
    ens_df = ens_df.drop(['bearing'], axis=1)
    return ens_df

def vel_cal(df):
    df = df.groupby(['ens'], group_keys=True).apply(dir_vel_zdist)
    return df

def process_line(line, masked_tl):
    # Process transect file for plotting
    # Tide correction
    line = pd.merge_asof(line, masked_tl, on='date', direction = 'backward')
    line['hn'] = line['h'] - line['Height']
    line['hbn'] = line['hb'] - line['Height']
    
    # Recompute velocity, bearing and add Z distance
    line = vel_cal(line)
    
    # remove duplicate data for recompute distance calculation
    df2 = line.drop_duplicates(subset = 'ens')
    df2.reset_index(inplace=True, drop=True)

    # define spatial geometry
    gdf = gpd.GeoDataFrame(df2, geometry=gpd.points_from_xy(df2.lon, df2.lat),
                           crs = 'EPSG:4326').to_crs('EPSG:32750')
    lx = LineString(gdf['geometry'].to_list())
    
    # calculate line bearing orientation need to reveresed or not
    xx =  lx.coords
    p1 = xx[0]
    p2 = xx[-1]

    dx = p2[0]-p1[0]
    dy = p2[1]-p1[1]
    angle = degrees(atan(dx/dy))

    if (dx>0 and dy>0):
        plot_brg = angle
    elif ((dx>0 and dy<0) or (dx<0 and dy<0)):
        plot_brg = (angle+180) %360
    else:
        plot_brg = (angle+360) %360
    
    
    
    # recalculate distance by projected coordinate
    gdf['dist_prev'] = 0
    gdf['dist_tot'] = 0
    for i in gdf.index[:-1]:
        gdf.loc[i+1, 'dist_prev'] = gdf.loc[i, 'geometry'].distance(gdf.loc[i+1, 'geometry'])
        
    for j in gdf.index[:-1]:
        gdf.loc[j+1, 'dist_tot'] = gdf.loc[j, 'dist_tot'] + gdf.loc[j+1, 'dist_prev']

    # round distance
    gdf.dist_tot = round(gdf.dist_tot,3)

    #replace with new distance
    line['dist'] = line['dist'].map(dict(zip(gdf.dist, gdf.dist_tot)))
    
    # create pivot table
    arr = line.pivot_table(index='hb', columns='dist', values='vel')
    
    # save df
    out = OUTPATH / "Transect_file"
    out.mkdir(parents=True, exist_ok=True) 
    file_out = re.sub(r'(?i).txt', 'test.csv', fo)
    
    if not out.exists():
        out.mkdir(parents=True)
        
    line.to_csv(out / file_out, encoding='utf-8')
    return line, gdf, arr, plot_brg

def save_shp(file, geodata):
    # save data to SHP
    # define folder for SHP results
    outshp = OUTPATH / "Transect_ESRI_Shapefile"
    outshp.mkdir(parents=True, exist_ok=True)
    shp_out = re.sub(r'(?i).txt', '.shp', fo)
    
    if not outshp.exists():
        outshp.mkdir(parents=True)
    
    # create line geometry
    line_geom = LineString(geodata['geometry'].to_list())
    line_name = Path(file).stem
    line_num = re.findall(r'\d+', line_name)
    trans_line = gpd.GeoDataFrame({'geometry':[line_geom], 'name':["T_" + "".join(line_num)]}, 
                                  crs='EPSG:32750').to_crs('EPSG:4326')
    trans_line.to_file(outshp / shp_out, encoding='utf-8')
    return print(f'* Finished save to shapefile {shp_out}')

def label_orient(line_heading):
    # SETTING PLOT'S LABEL BASED ON LINE ORIENTATION
    # define label line orientation. plot_brg is line heading / label EOL
    label_dict = {
        (337.5 < line_heading <= 22.5): ('S', 'N'),
        (22.5 < line_heading <= 67.5): ('SW', 'NE'),
        (67.5 < line_heading <= 122.5): ('W', 'E'),
        (122.5 < line_heading <= 157.5): ('NW', 'SE'),
        (157.5 < line_heading <= 202.5): ('N', 'S'),
        (202.5 < line_heading <= 247.5): ('NE', 'SW'),
        (247.5 < line_heading <= 292.5): ('E', 'W'),
        (292.5 < line_heading <= 337.5): ('SE', 'NW'),
    }
    lbl_start, lbl_end = label_dict[True]
    return lbl_start, lbl_end
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path # Assurez-vous que Path est bien importé

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# =========================================================================
# ====> REMPLACEZ VOTRE ANCIENNE FONCTION PAR CELLE-CI <====
# =========================================================================
def analyze_transect_zones(df, gdf, file_name_base, output_folder, dist_div, depth_div):
    """
    Version FINALE et CORRIGÉE : Accepte 'gdf', utilise 'h' pour la profondeur
    de la mesure, et 'gdf' pour le fond. Gère les zones vides.
    """
    print("\n* Analyse des vitesses moyennes par zones relatives (Logique Corrigée)...")

    df_copy = df.copy()
    
    # --- CORRECTION DE LOGIQUE ---
    # La profondeur de la mesure est la colonne 'h'.
    # La profondeur du fond est dans 'gdf'.
    depth_col = 'z'

    # --- 1. Normalisation ---
    dist_min, dist_max = df_copy['dist'].min(), df_copy['dist'].max()
    df_copy['dist_norm'] = (df_copy['dist'] - dist_min) / (dist_max - dist_min)
    
    # Utiliser gdf pour interpoler le fond marin (méthode robuste)
    seabed_depth_at_points = np.interp(df_copy['dist'], gdf['dist_tot'], gdf['z'])
    df_copy['depth_norm'] = df_copy[depth_col] / seabed_depth_at_points
    df_copy = df_copy[df_copy['depth_norm'] <= 1.0].copy()

    # --- 2. Assignation des zones ---
    dist_labels = [f"Dist {int(dist_div[i]*100)}-{int(dist_div[i+1]*100)}%" for i in range(len(dist_div)-1)]
    depth_labels = [f"Prof {int(depth_div[i]*100)}-{int(depth_div[i+1]*100)}%" for i in range(len(depth_div)-1)]
    df_copy['dist_zone'] = pd.cut(df_copy['dist_norm'], bins=dist_div, labels=dist_labels, include_lowest=True)
    df_copy['depth_zone'] = pd.cut(df_copy['depth_norm'], bins=depth_div, labels=depth_labels, include_lowest=True)

    # --- 3. Calcul de la moyenne par zone ---
    df_copy['zone_id'] = df_copy['depth_zone'].astype(str) + " / " + df_copy['dist_zone'].astype(str)
    zone_means = df_copy.groupby('zone_id')['vel'].mean().reset_index()
    zone_means.rename(columns={'vel': 'vitesse_moyenne_ms'}, inplace=True)
    
    print("Résultats de l'analyse par zone :")
    print(zone_means)
    
    # --- 4. Visualisation ---
    num_dist_zones = len(dist_labels)
    num_depth_zones = len(depth_labels)
    
    mean_vel_matrix = np.full((num_depth_zones, num_dist_zones), np.nan)
    
    for i, d_label in enumerate(depth_labels):
        for j, h_label in enumerate(dist_labels):
            zone_name = f"{d_label} / {h_label}"
            value = zone_means[zone_means['zone_id'] == zone_name]['vitesse_moyenne_ms']
            if not value.empty:
                mean_vel_matrix[i, j] = value.iloc[0]

    fig, ax = plt.subplots(figsize=(8, 4))
    cmap = plt.cm.jet
    cmap.set_bad(color='grey', alpha=0.5)
    im = ax.imshow(mean_vel_matrix, cmap=cmap, aspect='auto', vmin=0, vmax=max(1.0, zone_means['vitesse_moyenne_ms'].max() if not zone_means.empty else 1.0))

    for i in range(num_depth_zones):
        for j in range(num_dist_zones):
            value = mean_vel_matrix[i, j]
            if np.isnan(value):
                text_to_display = "N/A"
            else:
                text_to_display = f"{value:.2f} m/s"
            ax.text(j, i, text_to_display, ha="center", va="center", color="white", fontweight="bold")

    ax.set_xticks(np.arange(num_dist_zones))
    ax.set_yticks(np.arange(num_depth_zones))
    ax.set_xticklabels(dist_labels)
    ax.set_yticklabels(depth_labels)
    
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    fig.suptitle(f"Analyse par Zones - {file_name_base}", fontsize=14)
    fig.tight_layout()

    # Sauvegarde
    output_folder.mkdir(parents=True, exist_ok=True)
    output_image_path = output_folder / f"{file_name_base}_zone_analysis.png"
    output_csv_path = output_folder / f"{file_name_base}_zone_results.csv"
    
    fig.savefig(output_image_path, dpi=150, bbox_inches='tight')
    zone_means.to_csv(output_csv_path, index=False)
    
    print(f"-> Graphique d'analyse des zones sauvegardé : {output_image_path}")
    print(f"-> Résultats des zones sauvegardés : {output_csv_path}")
    
    plt.close(fig)
    return zone_means
def transect_proc(file_in, ftide):
    # process line transect
    fo = Path(file_in).name
    # Read transect file and tide file
    tide = pd.read_csv(ftide, parse_dates=['Time'], na_values=['*****'])

    line = pd.read_csv(file_in, sep='\t', skiprows=33, parse_dates=['date'])

    # Process tide file
    t_sol, t_eol, masked_tl, tidenew, xc, yc = process_tide(line, tide)
    
    # Process transect file
    df, gdf, arr, plot_brg = process_line(line, masked_tl)

    # Process averaging velocity every 5m depth
    av = avg_5m.cal_avg_speed(line)

    # get midle time of tide every transect
    #sum_dat = {'name':[f'T_{fo[9:11]}'], 'x':[xc], 'y':[yc]}
    #summary = pd.DataFrame(sum_dat)
    #summary.to_csv('overview_tide.txt', mode='a', index=False, header=False)

    # clean direction
    dfc = currentprop.clean_dir(df)
    
    # Calculate shear velocity
    shear_raw = currentprop.shear(dfc, fo)

    # process to DAV calculation
    dav_dat = dav.proc_dav(shear_raw, fo)
   
    # Plot DAV feather plot
    dav.davplot(dav_dat, fo)
    
    # save dav to shp, average value every 75 ensemble for visualize purpose
    dav.dav_to_shp(dav_dat, 75, fo)
    
    # calculate turbulence intensity
    turb = currentprop.turbulence(shear_raw, plot_brg, fo)
                
    # save to shp
    save_shp(file_in, gdf)
    
    # define label orientation
    lbl_start, lbl_end = label_orient(plot_brg)
    
    # plot turbulence intensity
    currentprop.turb_plot(turb, gdf, fo, lbl_start, lbl_end)
    # ... dans la fonction transect_proc ...


    
    # =========================================================================
    # ====> AJOUTEZ CE BLOC DE CODE D'APPEL ICI <====
    # =========================================================================
    
    # --- Début de l'analyse par zones ---
    
    # 1. Définissez comment vous voulez diviser le transect
    #    Ici, on divise en 6 zones (3 horizontales x 2 verticales)
    distance_divisions = [0, 1/3, 2/3, 1.0]
    profondeur_divisions = [0, 1/3, 2/3, 1.0]
    
    # 2. Définissez où sauvegarder les résultats de cette nouvelle analyse
    output_folder_zones = OUTPATH / "Analyse_Zones"
    output_folder_zones.mkdir(parents=True, exist_ok=True)

    file_name_base = Path(file_in).stem # Nom du fichier sans extension, ex: 'transect_23-03-03'
    
    # L'appel dans transect_proc
    resultats_par_zone = analyze_transect_zones(
        df=df,                          
        gdf=gdf,                        # L'argument 'gdf' est bien là
        file_name_base=file_name_base,  
        output_folder=output_folder_zones, 
        dist_div=distance_divisions,    
        depth_div=profondeur_divisions  
    )
    # --- Fin de l'analyse par zones ---
    
    # =========================================================================
    


    
    """ Plotting"""
    # define image folder
    outpath = OUTPATH / "Transect_Image_Profile"
    outpath.mkdir(parents=True, exist_ok=True)

    file_out = re.sub(r'(?i).txt', '_avg5m.jpg', fo)
    
    if not outpath.exists():
        outpath.mkdir(parents=True)
    
    # define image name
    linename = Path(file_in).stem
    pattern = r'(\d{2}-\d{2}-\d{2})'
    match = re.search(pattern, linename)
    lineid = match.group(1)
    plt_name = f'Transect {lineid}'
    
    # data plotting
    fig, ax = plt.subplots(ncols=2, 
                           gridspec_kw={'width_ratios': [3, 1], 'wspace': 0.15})
    fig.suptitle(plt_name + " - (" + t_sol.strftime('%d %B %Y') + ")")

    # TRANSECT PLOT ver.1 avg 5m
    # plot velocity magnitude and invert Y-axis
    vel = av['vel']
    u = av['u5']
    v = av['v5']

    norm = colors.Normalize(vmin=0.1, vmax=1)
    Q = ax[0].quiver(av['dist'], av['ens_h'], u, v, vel, width=0.0008, cmap='jet', norm=norm, scale=1 / 0.005)
    ax[0].quiverkey(Q, 0.15, 0.91, 5, r'East 5 m/s', labelpos='E',coordinates='figure')
    ax[0].invert_yaxis()

    # plot MSL & seabed profile 
    ax[0].axhline(y=0, color='k', lw=1, linestyle='--')
    ax[0].plot(gdf.dist_tot, gdf.Height*-1,'r')
    
    zmax = av.h.max() +5
    ax[0].plot(av.dist, av.h,'k')
    ax[0].fill_between(av.dist, av.h, zmax, color='slategrey')
    dist_max = gdf.dist_tot.max()
    # setting label
    bbox = dict(facecolor='none', edgecolor='black')
    ax[0].annotate(lbl_start, xy=(0, 0), xycoords='axes fraction',xytext=(-0.01, -0.25), bbox = bbox)
    ax[0].annotate(lbl_end, xy=(0, 0), xycoords='axes fraction',xytext=(0.99, -0.25),bbox = bbox)
    ax[0].tick_params(axis="x", direction='in', length=8)
    ax[0].set_ylim(zmax, -5.0)
    
    ax[0].set_xlim(-50,dist_max)
    ax[0].set_xlabel('Distance (m)')
    ax[0].set_ylabel('Depth (m)')

    # define colorbar properties
    axins = inset_axes(ax[0], width="80%", height="5%",loc='lower center', borderpad=-5)

    # plot colorbar
    fig.colorbar(Q, cax=axins, orientation="horizontal", label='Velocity m/s')

    # Tide Plot
    # plot tide height
    ax[1].plot(tidenew.index, tidenew.Height, color='deepskyblue')
    ax[1].plot(masked_tl.index, masked_tl.Height, color='red')
    ax[1].plot(xc, yc, marker='o', markersize=5, color='red',label='Survey Time')
    ax[1].axhline(y=0, color='g', lw=0.75)

    ax[1].legend(loc='lower right')

    for label in ax[1].get_xticklabels():
        label.set_rotation(40)
        label.set_horizontalalignment('right')
    
    ax[1].set_xlabel('Time (UTC)')
    ax[1].set_ylabel('Height (m)')
    ax[1].grid(lw=0.3)

    # Save image
    fig.savefig(outpath / file_out, bbox_inches='tight', dpi=300)
    plt.close()
    print(f"-> {file_out} image saved")

    print(f'Process file {fo}, DONE! \n')
    return resultats_par_zone, t_sol
# ... (après la fonction transect_proc) ...

# =========================================================================
# ====> NOUVELLE FONCTION POUR L'ANALYSE COMPARATIVE <====
# =========================================================================
def compare_all_transects(all_results_list):
    """
    Analyse les résultats de plusieurs transects pour comparer l'évolution des vitesses.
    Version CORRIGÉE : trace les graphiques même s'il n'y a qu'un seul point de donnée.
    """
    print("\n" + "="*50)
    print(" DÉBUT DE L'ANALYSE COMPARATIVE DE TOUS LES TRANSECTS")
    print("="*50)

    if not all_results_list:
        print("-> Aucune donnée de transect n'a été collectée. Analyse comparative annulée.")
        return

    # --- 1. Préparation des données ---
    full_df_list = []
    for result in all_results_list:
        if result['results_df'].empty:
            print(f"   - Fichier {result['file_name']} n'a produit aucun résultat de zone, il sera ignoré.")
            continue
        temp_df = result['results_df'].copy()
        temp_df['start_time'] = result['start_time']
        full_df_list.append(temp_df)
    
    if not full_df_list:
        print("-> Aucun des fichiers traités n'avait de données de zone valides. Analyse terminée.")
        return
        
    comparison_df = pd.concat(full_df_list, ignore_index=True)

    # Créer le dossier pour les résultats globaux
    output_dir = OUTPATH / "Analyse_Globale"
    output_dir.mkdir(parents=True, exist_ok=True)
    zone_plot_dir = output_dir / "graphiques_evolution_par_zone"
    zone_plot_dir.mkdir(exist_ok=True)

    output_dir.mkdir(parents=True, exist_ok=True)

    # --- 2. Création du tableau comparatif ---
    summary_table = comparison_df.pivot_table(
        index='zone_id', 
        columns='start_time', 
        values='vitesse_moyenne_ms'
    ).round(3)
    summary_table.columns = [col.strftime('%Y-%m-%d %H:%M') for col in summary_table.columns]
    
    summary_csv_path = output_dir / 'comparaison_vitesses_par_zone.csv'
    summary_table.to_csv(summary_csv_path)
    print(f"\n-> Tableau comparatif des vitesses sauvegardé : {summary_csv_path}")
    print("Aperçu du tableau :")
    print(summary_table)

    # --- 3. Création des graphiques d'évolution par zone ---
    print("\n-> Génération des graphiques d'évolution par zone...")
    zone_plot_dir = output_dir / 'graphiques_evolution_par_zone'
    zone_plot_dir.mkdir(exist_ok=True)

    unique_zones = comparison_df['zone_id'].unique()

    for zone in unique_zones:
        # Isoler les données pour la zone actuelle
        zone_data = comparison_df[comparison_df['zone_id'] == zone].sort_values('start_time')
        
        # --- CORRECTION DE LOGIQUE ICI ---
        if zone_data.empty:
            continue # Ne rien faire si la zone est vide (ne devrait pas arriver, mais sécurité)

        fig, ax = plt.subplots(figsize=(10, 5))
        
        # Si on a 2 points ou plus, on trace une ligne et des marqueurs
        if len(zone_data) >= 2:
            ax.plot(zone_data['start_time'], zone_data['vitesse_moyenne_ms'], marker='o', linestyle='-')
        # Si on a EXACTEMENT 1 point, on trace juste le point (scatter plot)
        elif len(zone_data) == 1:
            ax.scatter(zone_data['start_time'], zone_data['vitesse_moyenne_ms'], marker='o', s=50) # s=50 pour taille du point
            print(f"   - Zone '{zone}' n'a qu'un seul point de donnée (graphique 'point seul' créé).")

        # Mise en forme du graphique
        ax.set_title(f"Évolution de la Vitesse Moyenne\nZone: {zone}", fontsize=14)
        ax.set_xlabel("Date et Heure du Transect", fontsize=10)
        ax.set_ylabel("Vitesse Moyenne (m/s)", fontsize=10)
        ax.grid(True, which='both', linestyle='--', linewidth=0.5)
        fig.autofmt_xdate() # Améliore l'affichage des dates
        
        # S'assurer que l'axe Y a une plage raisonnable
        if not zone_data.empty:
            min_val = zone_data['vitesse_moyenne_ms'].min()
            max_val = zone_data['vitesse_moyenne_ms'].max()
            padding = max(0.1, (max_val - min_val) * 0.2) # Ajoute un peu de marge
            ax.set_ylim(min_val - padding, max_val + padding)


        # Sauvegarde du graphique
        safe_zone_name = zone.replace('/', '_').replace('%', '').replace(' ', '_').replace('-', '_')
        plot_path = zone_plot_dir / f"evolution_{safe_zone_name}.png"
        fig.savefig(plot_path, dpi=150, bbox_inches='tight')
        plt.close(fig)

    print(f"-> {len(unique_zones)} graphiques d'évolution sauvegardés dans : {zone_plot_dir}")
    print("\n" + "="*50)
    print(" ANALYSE COMPARATIVE TERMINÉE")
    print("="*50)
# ... (votre code existant jusqu'au bloc principal) ...
if __name__ == "__main__":

    # Utilisation de chemins absolus et robustes
    os.environ["ADCP_OUTPATH"] = str(Path(__file__).resolve().parent / "Orde_1 - CP_PRODUCT")
    script_dir = Path(__file__).resolve().parent
    transect_folder = script_dir / "Orde_1 - CP_PRODUCT/Transect_file"
    tide_folder = script_dir / "Orde_1 - CP_PRODUCT"

    transect_file = glob.glob(str(transect_folder / "*.txt"))
    tide_file = glob.glob(str(tide_folder / "tide_*.csv"))
        
    if not tide_file:
        print(f"ERREUR : Aucun fichier de marée (ex: tide_xxx.csv) trouvé dans : {tide_folder}")
        exit()
    
    ftd = tide_file[0]
    print(f"Fichier de marée détecté : {ftd}")
        
    if not transect_file:
        print(f"ERREUR : Aucun fichier transect (*.txt) trouvé dans : {transect_folder}")
        exit()
    else:
        print("\nFichiers transects détectés pour l'analyse : \n", "\n".join(transect_file))
        
        all_transect_results = []

        for file_in in transect_file:
            fo = Path(file_in).name
            
            try:
                # On recharge le fichier transect en ignorant les 33 lignes d'en-tête
                df_raw = pd.read_csv(file_in, sep="\t", skiprows=33)

                if "date" not in df_raw.columns:
                    print(f"   -> AVERTISSEMENT : La colonne 'date' est absente dans '{fo}'. Fichier ignoré.")
                    continue

                # Conversion robuste de la colonne date
                df_raw["date"] = pd.to_datetime(
                    df_raw["date"], format="%d.%m.%Y %H:%M:%S", errors="coerce"
                )

                transect_date = df_raw["date"].min()
                if pd.isna(transect_date):
                    print(f"   -> AVERTISSEMENT : Impossible de convertir les dates du fichier '{fo}'. Fichier ignoré.")
                    continue

                # Passage dans ton traitement ADCP
                results_df, internal_start_time = transect_proc(file_in, ftd)

                # Ajout aux résultats pour comparaison
                all_transect_results.append({
                    "file_name": fo,
                    "start_time": transect_date,  # <--- date trouvée dans dataset
                    "results_df": results_df
                })

            except Exception as e:
                print(f"\n!!! ERREUR critique lors du traitement du fichier {fo}: {e}")
                print("--- Le script passe au fichier suivant. ---")
                continue
        
        # Lancement de la comparaison finale
        if all_transect_results:
            compare_all_transects(all_transect_results)
        else:
            print("\nAucun transect n'a pu être traité avec succès. Analyse comparative annulée.")
