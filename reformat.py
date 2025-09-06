# Contenu de reformat.py

from pathlib import Path
from datetime import datetime as dt
import glob
import re
from pandas import DataFrame, concat
from numpy import nan
from itertools import islice
import os

# --- Début des fonctions (inchangées) ---

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
    
    df['bs'] = df[['bs1', 'bs2', 'bs3', 'bs4']].mean(axis=1).round(3)
    df = df.replace([-32768, 2147483647, 255], nan)
    df.drop(['percgood', 'q', 'bs1', 'bs2', 'bs3', 'bs4'], inplace=True, axis=1)
    df['date'] = date
    df['ens'] = ensnum
    df['dist'] = ensdist
    df['h'] = ensh
    df['lat'] = enslat
    df['lon'] = enslon
    df['roll'] = ensroll
    df['pitch'] = enspitch
    df[['vel', 'u', 'v', 'w', 'errv']] = (df[['vel', 'u', 'v', 'w', 'errv']] * 0.01).round(3)
    df = df.dropna()
    res = df[['date', 'ens', 'dist', 'lat', 'lon', 'roll', 'pitch', 'h', 'hb', 'u', 'v', 'w', 'errv', 'vel', 'dir', 'bs']]
    return res

def file_proc(path_in, path_out_folder, file_out_name, mt):
    out_path_full = path_out_folder / file_out_name
    with open(path_in, "r") as fn:
        print(f'Process file {path_in}, Please Wait!')
        fn.readline(); fn.readline(); fn.readline() # skip three empty lines
        
        df_list = [] # Use a list for performance
        head = get_chunk(fn, 6)
        while head:
            opr = head_proc(head)
            chunk = get_chunk(fn, opr['nbins'])
            ens = ens_proc(chunk, opr['date'], opr['ens'], opr['b'], opr['h'], opr['lat'], opr['lon'], opr['roll'], opr['pitch'])
            df_list.append(ens)
            head = get_chunk(fn, 6)
        
        if not df_list:
            print(f"Aucune donnée valide trouvée dans {path_in}. Fichier de sortie non créé.")
            return

        df = concat(df_list, ignore_index=True)
        df = df.dropna()
        
        df.to_csv(out_path_full, sep='\t', index=False, na_rep='-32768')
        print(f'Finished for {file_out_name}')
        
        if mt:
            with open(mt) as fp:
                data = fp.read()
            with open(out_path_full) as fp:
                data2 = fp.read()
            
            data += data2
      
            with open (out_path_full, 'w') as fp:
                fp.write(data)
                print(f'Finished add Metadata to {file_out_name}\n')


# --- NOUVELLE FONCTION PRINCIPALE ---
# On a mis toute la logique de recherche de fichiers dans une fonction
# pour que app.py puisse l'appeler.

def run_reformat_process():
    """
    Fonction principale qui recherche les fichiers RAW et lance le reformatage.
    Retourne une liste de messages pour l'affichage dans Streamlit.
    """
    logs = [] # On va stocker les messages ici
    script_dir = Path(__file__).resolve().parent
    
    outpath = script_dir / 'Orde_1 - CP_PRODUCT/Transect_file'
    raw_files_folder = script_dir / "Orde_0 - CP_RAW/PLAYBACK DATA"
    metadata_path = script_dir / "Orde_1 - CP_PRODUCT/metadata.txt"

    outpath.mkdir(parents=True, exist_ok=True)
    
    mtdt_files = glob.glob(str(metadata_path))
    mt = mtdt_files[0] if mtdt_files else None

    pattern = f"{raw_files_folder}/*_ASC.[Tt][Xx][Tt]"
    raw_files = glob.glob(pattern)
    
    if not raw_files:
        log_message = (f'Aucun fichier correspondant au modèle "*_ASC.txt" n\'a été trouvé dans le dossier :\n'
                       f' -> {raw_files_folder}')
        logs.append(log_message)
        return logs

    logs.append("Fichiers ASCII détectés pour le reformatage :")
    for f in raw_files:
        logs.append(f"- {Path(f).name}")
    
    for f in raw_files:
        file_in = f
        fo = Path(file_in).name
        file_out_name = re.sub(r'(?i)_ASC\.[Tt][Xx][Tt]', '.txt', fo)
        
        try:
            file_proc(file_in, outpath, file_out_name, mt)
            logs.append(f"✅ Traitement de {fo} terminé avec succès.")
        except Exception as e:
            log_message = (f"❌ ERREUR critique lors du traitement du fichier {fo}: {e}\n"
                           f"--- Le script passe au fichier suivant. ---")
            logs.append(log_message)
            
    return logs


# Ce bloc ne s'exécute que si on lance `python reformat.py` directement
# Il ne s'exécutera PAS quand Streamlit l'importera.
if __name__ == "__main__":
    run_reformat_process()
