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
        interval_depth = 5
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

if __name__ == "__main__":

    transect_file = glob.glob("Orde_1 - CP_PRODUCT/Transect_file/*.txt")
     
    if not transect_file:
        print('No files to convert.')
        exit()
    else:
        print("Detected ASCII Transect files: \n", "\n".join(transect_file))
        for file_in in transect_file:
            fo = Path(file_in).name
            avg_5m(file_in)