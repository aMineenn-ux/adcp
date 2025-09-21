import pandas as pd
import numpy as np
import geopandas as gpd
import glob
from pathlib import Path
import re

def dir_vel(ens_df):
    # calculate velocity and direction from vector
    ens_df['vel'] = np.sqrt((ens_df['u5']**2)+(ens_df['v5']**2)).round(3)
    ens_df['bearing'] = np.degrees(np.arctan(ens_df['u5']/ens_df['v5'])).round(3)
    ens_df.loc[(ens_df['u5']>0) & (ens_df['v5']>0), 'dir'] = ens_df['bearing']
    ens_df.loc[((ens_df['u5']>0) & (ens_df['v5']<0)) | ((ens_df['u5']<0) & (ens_df['v5']<0)), 'dir'] = (ens_df['bearing']+180) %360
    ens_df.loc[(ens_df['u5']<0) & (ens_df['v5']>0), 'dir'] = (ens_df['bearing']+360) %360
    return ens_df

def cal_avg_speed(df):
        # remove duplicate data for recompute distance calculation
    df2 = df.drop_duplicates(subset = 'ens')
    df2.reset_index(inplace=True, drop=True)

    # define spatial geometry
    gdf = gpd.GeoDataFrame(df2, geometry=gpd.points_from_xy(df2.lon, df2.lat),
                           crs = 'EPSG:4326').to_crs('EPSG:32750')
        
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
    df['dist'] = df['dist'].map(dict(zip(gdf.dist, gdf.dist_tot)))
    
    df = df.reset_index(drop=True)
    dfl = df.groupby('ens')
    dum = []

    for ens, ens_df in dfl:
        depth_values = ens_df['hb'].round(1)
        u_values = ens_df['u']
        v_values = ens_df['v']
        interval_depth = 0.1
        interval_count = int(max(depth_values) / interval_depth)
        interval_u_avg = [0] * interval_count
        interval_v_avg = [0] * interval_count

        for i in range(interval_count):
            start_depth = i * interval_depth
            end_depth = (i + 1) * interval_depth

            interval_u_sum = 0
            interval_v_sum = 0
            count = 0

            for depth, u, v in zip(depth_values, u_values, v_values):
                if start_depth < depth <= end_depth:
                    interval_u_sum += u
                    interval_v_sum += v
                    count += 1

            if count > 0:
                interval_u_avg[i] = interval_u_sum / count
                interval_v_avg[i] = interval_v_sum / count

        fifth_depths = [(i + 1) * interval_depth for i in range(interval_count)]
        avg_speed_df = pd.DataFrame({'ens': [ens] * interval_count, 
                                     'date': ens_df['date'].iloc[1],
                                     'dist': ens_df['dist'].iloc[1],
                                     'lat': ens_df['lat'].iloc[1],
                                     'lon': ens_df['lon'].iloc[1],
                                     'h': ens_df['h'].iloc[1],
                                     'u5': interval_u_avg, 
                                     'v5': interval_v_avg, 
                                     'ens_h': fifth_depths})
        dum.append(avg_speed_df)

    avg_speed = pd.concat(dum, ignore_index=True)
    dir_vel(avg_speed)
    avg_speed = avg_speed.drop(['bearing'], axis=1)
    return avg_speed

# NOUVELLE FONCTION pour le backscatter 'bs'
def cal_avg_backscatter(df):
    """
    Calcule le backscatter (bs) moyen par tranche de 5m.
    Le style est identique à cal_avg_speed.
    """
    # La partie sur la distance est identique et peut être réutilisée
    df2 = df.drop_duplicates(subset = 'ens')
    df2.reset_index(inplace=True, drop=True)
    gdf = gpd.GeoDataFrame(df2, geometry=gpd.points_from_xy(df2.lon, df2.lat),
                           crs = 'EPSG:4326').to_crs('EPSG:32750')
    gdf['dist_prev'] = 0
    gdf['dist_tot'] = 0
    for i in gdf.index[:-1]:
        gdf.loc[i+1, 'dist_prev'] = gdf.loc[i, 'geometry'].distance(gdf.loc[i+1, 'geometry'])
    for j in gdf.index[:-1]:
        gdf.loc[j+1, 'dist_tot'] = gdf.loc[j, 'dist_tot'] + gdf.loc[j+1, 'dist_prev']
    gdf.dist_tot = round(gdf.dist_tot,3)
    # Dans la ligne ci-dessous, j'ai remplacé 'dist' par 'ens' dans zip pour plus de robustesse
    df['dist'] = df['ens'].map(dict(zip(gdf.ens, gdf.dist_tot)))
    
    df = df.reset_index(drop=True)
    dfl = df.groupby('ens')
    dum = []

    # La boucle de moyennage est adaptée pour 'bs'
    for ens, ens_df in dfl:
        depth_values = ens_df['hb'].round(1)
        bs_values = ens_df['bs'] # On récupère les valeurs de 'bs'
        interval_depth = 5
        
        if depth_values.empty or depth_values.isnull().all():
            continue # On ignore les ensembles sans données de profondeur
        
        interval_count = int(max(depth_values) / interval_depth)
        interval_bs_avg = [np.nan] * interval_count # Liste pour stocker les moyennes de 'bs'

        for i in range(interval_count):
            start_depth = i * interval_depth
            end_depth = (i + 1) * interval_depth

            interval_bs_sum = 0 # Somme pour 'bs'
            count = 0

            # On itère sur la profondeur et 'bs'
            for depth, bs in zip(depth_values, bs_values):
                if start_depth < depth <= end_depth:
                    # On s'assure que bs n'est pas une valeur nulle (NaN)
                    if pd.notna(bs):
                        interval_bs_sum += bs
                        count += 1

            if count > 0:
                interval_bs_avg[i] = interval_bs_sum / count

        # Profondeur au centre de l'intervalle pour un meilleur plotting
        fifth_depths = [(i + 0.5) * interval_depth for i in range(interval_count)]
        avg_bs_df = pd.DataFrame({'ens': [ens] * interval_count, 
                                     'date': ens_df['date'].iloc[0],
                                     'dist': ens_df['dist'].iloc[0],
                                     'lat': ens_df['lat'].iloc[0],
                                     'lon': ens_df['lon'].iloc[0],
                                     'h': ens_df['h'].iloc[0],
                                     'bs_avg': interval_bs_avg, # Colonne résultat
                                     'ens_h': fifth_depths})
        dum.append(avg_bs_df)
    
    if not dum:
        return pd.DataFrame() # Retourne un dataframe vide si rien n'a été traité

    avg_bs = pd.concat(dum, ignore_index=True).dropna(subset=['bs_avg'])
    # Pas besoin d'appeler dir_vel car nous n'avons pas de vecteurs u/v
    return avg_bs
def avg_5m(file_in):
    print(f'Process file {fo}, Please Wait!')
    outavg = Path('Orde_1 - CP_PRODUCT/Transect_file/Transect_average')
    outavg.mkdir(parents=True, exist_ok=True)
    avg_out = re.sub(r'(?i).txt', '_avg.csv', fo)

    line = pd.read_csv(file_in, sep='\t', skiprows=33, parse_dates=['date'])
    average = cal_avg_speed(line)

    average.to_csv(outavg / avg_out, index=False)
    print(f'* Finished save file {avg_out}')
    return 
# NOUVELLE FONCTION principale pour le BACKSCATTER
def avg_5m_bs(file_in):
    """
    Fonction principale pour traiter le BACKSCATTER.
    """
    fo = Path(file_in).name
    print(f'Process BACKSCATTER pour le fichier {fo}, Veuillez patienter !')
    # On sauvegarde dans un dossier différent pour ne pas mélanger les fichiers
    outavg = Path('Orde_1 - CP_PRODUCT/Transect_file/Transect_average_bs') 
    outavg.mkdir(parents=True, exist_ok=True)
    # On utilise un suffixe de fichier différent
    avg_out = re.sub(r'(?i)\.txt$', '_bs_avg.csv', fo)

    try:
        line = pd.read_csv(file_in, sep='\t', skiprows=33, parse_dates=['date'])
        
        # Vérification cruciale : la colonne 'bs' existe-t-elle ?
        if 'bs' not in line.columns:
            print(f"! ATTENTION: Colonne 'bs' non trouvée dans le fichier {fo}. Traitement du backscatter ignoré.")
            return

        average_bs = cal_avg_backscatter(line)
        
        if not average_bs.empty:
            average_bs.to_csv(outavg / avg_out, index=False, float_format='%.3f')
            print(f'* Fichier de backscatter sauvegardé : {avg_out}')
        else:
            print(f'* Pas de données de backscatter à traiter pour {fo}')

    except Exception as e:
        print(f"ERREUR lors du traitement du backscatter pour {fo}: {e}")
if __name__ == "__main__":

    transect_file = glob.glob("Orde_1 - CP_PRODUCT/Transect_file/*.txt")
     
    if not transect_file:
        print('Aucun fichier à convertir.')
        exit()
    else:
        print("Fichiers de transect ASCII détectés : \n", "\n".join(transect_file))
        for file_in in transect_file:
            fo = Path(file_in).name # Déplacé ici pour n'être calculé qu'une fois
            print("-" * 50)
            
            # Appel de la fonction originale pour la vitesse
            avg_5m(file_in) 
            
            # NOUVEL APPEL : Lancement du traitement pour le backscatter
            avg_5m_bs(file_in)

        print("-" * 50)
        print("Traitement de tous les fichiers terminé.")
