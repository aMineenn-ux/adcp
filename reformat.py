#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on August 08 22:29:24 2023
Add command to extract Pitch and Roll Value

@author: sendy

modified from dopler.py

original file from https://github.com/esmoreido/dopler
"""

from pathlib import Path
from datetime import datetime as dt
import glob
import re
from pandas import DataFrame, concat
from numpy import nan
from itertools import islice
import os
OUTPATH = Path(os.environ.get("ADCP_OUTPATH", "Orde_1 - CP_PRODUCT"))
OUTPATH.mkdir(parents=True, exist_ok=True)

def get_chunk(file, n):
    return [x.strip() for x in islice(file, n)]

def head_proc(lines):
    avg = [float(lines[1].split()[8]),
           float(lines[1].split()[9]),
           float(lines[1].split()[10]),
           float(lines[1].split()[11]),
           ]   
    avg = DataFrame({'avg': avg}).replace([0, -32768], nan)
    avg = avg.mean().avg
    
    res = {'ens': int(lines[0].split()[7]),
           'date': dt.strftime(dt(year=2000 + int(lines[0].split()[0]), 
                                  month=int(lines[0].split()[1]),
                                  day=int(lines[0].split()[2]), 
                                  hour=int(lines[0].split()[3]),
                                  minute=int(lines[0].split()[4]), 
                                  second=int(lines[0].split()[5])),
                               format('%d.%m.%Y %H:%M:%S')),
           'h': round(avg, 3),
           'b': round(float(lines[2].split()[4]), 2),
           'lat': float(lines[3].split()[0]) if float(lines[3].split()[0]) != 30000. else nan,
           'lon': float(lines[3].split()[1]) if float(lines[3].split()[1]) != 30000. else nan,
           'nbins': int(lines[5].split()[0]),
           'roll': float(lines[0].split()[10]),
           'pitch': float(lines[0].split()[9]),
           }
   
    return res

def ens_proc(ens, date, ensnum, ensdist, ensh, enslat, enslon, ensroll, enspitch):
    df = DataFrame([x.split() for x in ens], 
                   columns=['hb', 'vel', 'dir', 'u', 'v', 'w', 'errv', 'bs1', 'bs2', 'bs3',
                            'bs4', 'percgood', 'q'], dtype='float')
    
    df['bs'] = df[['bs1', 'bs2', 'bs3', 'bs4']].mean(axis=1).round(3)  # calculate the average scatter
    df = df.replace([-32768, 2147483647, 255], nan)
    df.drop(['percgood', 'q', 'bs1', 'bs2', 'bs3', 'bs4'], inplace=True, axis=1)  # remove unnecessary scatter columns
    df['date'] = date
    df['ens'] = ensnum  # add the ensemble number
    df['dist'] = ensdist  # add distance from the edge
    df['h'] = ensh  # total depth
    df['lat'] = enslat  # latitude
    df['lon'] = enslon  # longitude
    df['roll'] = ensroll
    df['pitch'] = enspitch
    df[['vel', 'u', 'v', 'w', 'errv']] = (df[['vel', 'u', 'v', 'w', 'errv']] * 0.01).round(3)
    df = df.dropna()  # remove missing
    res = df[['date', 'ens', 'dist', 'lat', 'lon', 'roll', 'pitch', 'h', 'hb', 'u', 'v', 'w', 'errv', 'vel', 'dir', 'bs']]
    return res

def file_proc(path_in, path_out, mt):
    with open(path_in, "r") as fn:
        print(f'Process file {fo}, Please Wait!')
        fn.readline()  # skip three empty lines
        fn.readline()
        fn.readline()
        df = DataFrame()  # array for data
        # read the first piece of service information - always 6 lines
        head = get_chunk(fn, 6)
        while head:
            opr = head_proc(head)
            chunk = get_chunk(fn, opr['nbins'])
            ens = ens_proc(chunk, opr['date'], opr['ens'], opr['b'], opr['h'], opr['lat'], opr['lon'], opr['roll'], opr['pitch'])
            df = concat([df, ens], ignore_index=True)
            head = get_chunk(fn, 6)
        df = df.dropna()
        # save output
        df.to_csv(outpath / path_out, sep='\t', index=False, na_rep='-32768')
        print(f'Finished for {path_out}')
        
        # add metadata
        with open(mt) as fp:
            data = fp.read()
        with open(outpath / path_out) as fp:
            data2 = fp.read()
        
        data += data2
  
        with open (outpath / path_out, 'w') as fp:
            fp.write(data)
            print(f'Finished add Metadata to {path_out}\n')
    return

# =========================================================================
# ====> BLOC COMPLET À COPIER-COLLER À LA FIN DE reformat.py <====
# =========================================================================

if __name__ == "__main__":
    
    # On détermine le chemin absolu du script qui est en train de tourner
    # Cela rend le script portable et fiable, peu importe d'où il est lancé.
    script_dir = Path(__file__).resolve().parent
    
    # On construit les chemins des dossiers importants en se basant sur la localisation du script
    outpath = script_dir / 'Orde_1 - CP_PRODUCT/Transect_file'
    raw_files_folder = script_dir / "Orde_0 - CP_RAW/PLAYBACK DATA"
    metadata_path = script_dir / "Orde_1 - CP_PRODUCT/metadata.txt"

    # On s'assure que le dossier de sortie existe, sinon on le crée
    if not outpath.exists():
        outpath.mkdir(parents=True)

    # On utilise le chemin absolu pour chercher le fichier metadata
    mtdt = glob.glob(str(metadata_path))
    
    # --- Recherche des fichiers bruts ---
    # On utilise un pattern qui ignore la casse pour l'extension (.txt ou .TXT)
    # [Tt] signifie "T majuscule ou t minuscule".
    pattern = f"{raw_files_folder}/*_ASC.[Tt][Xx][Tt]"
    ff = glob.glob(pattern)
    
    # On vérifie si des fichiers ont été trouvés
    if not ff:
        print(f'Aucun fichier correspondant au modèle "*_ASC.txt" (insensible à la casse) n\'a été trouvé dans le dossier :')
        print(f' -> {raw_files_folder}')
        print('\nVeuillez vérifier :')
        print("1. Que le chemin d'accès est correct.")
        print("2. Que les fichiers sont bien présents dans ce dossier.")
        print("3. Que les noms de fichiers se terminent bien par '_ASC.txt' ou '_ASC.TXT'.")
        exit()
    else:
        print("Fichiers ASCII détectés pour le reformatage : \n", "\n".join(ff))
        
        # On boucle sur chaque fichier trouvé pour le traiter
        for f in ff:
            if not mtdt:
                print(f'Attention : Le fichier metadata.txt n\'a pas été trouvé. Le fichier {Path(f).name} sera créé sans en-tête de métadonnées.')
                mt = None 
            else:
                mt = mtdt[0]

            # Préparation des noms de fichiers
            file_in = f
            fo = Path(file_in).name
            
            # On renomme le fichier de sortie en remplaçant l'extension originale par un simple .txt
            file_out = re.sub(r'(?i)_ASC\.[Tt][Xx][Tt]', '.txt', fo)
            
            # On appelle la fonction de traitement pour le fichier en cours
            try:
                # La fonction file_proc est définie plus haut dans votre script
                file_proc(file_in, file_out, mt)
            except Exception as e:
                print(f"\n!!! ERREUR critique lors du traitement du fichier {fo}: {e}")
                print("--- Le script passe au fichier suivant. ---")