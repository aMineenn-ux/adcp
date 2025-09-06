import streamlit as st
from pathlib import Path
import subprocess
import zipfile
import shutil
import os
import pandas as pd
import sys

st.set_page_config(page_title="Pipeline ADCP", layout="wide")
st.title("Pipeline ADCP - Reformat et Analyse Comparative")

# -----------------------------
# 1️⃣ Upload fichiers RAW
# -----------------------------
raw_files = st.file_uploader(
    "Uploader vos fichiers RAW (_ASC.txt)", type="txt", accept_multiple_files=True
)

if st.button("Lancer tout le traitement"):

    if not raw_files:
        st.error("Il faut uploader au moins les fichiers RAW.")
        st.stop()
    else:
        st.info("Traitement en cours, cela peut prendre plusieurs minutes...")

        # -----------------------------
        # 2️⃣ Préparer le dossier de travail
        # -----------------------------
        work_dir = Path("streamlit_temp").resolve()
        if work_dir.exists():
            shutil.rmtree(work_dir)
        work_dir.mkdir(parents=True)

        raw_folder = work_dir / "Orde_0 - CP_RAW/PLAYBACK DATA"
        raw_folder.mkdir(parents=True, exist_ok=True)
        for f in raw_files:
            (raw_folder / f.name).write_bytes(f.read())

        # Fichiers fixes
        fixed_files_dir = Path("fixed_files")
        metadata_src = fixed_files_dir / "metadata.txt"
        tide_src = fixed_files_dir / "tide_alas.csv"
        output_dir = work_dir / "Orde_1 - CP_PRODUCT"
        output_dir.mkdir(parents=True, exist_ok=True)

        if metadata_src.exists():
            shutil.copy(metadata_src, output_dir / "metadata.txt")
        if tide_src.exists():
            shutil.copy(tide_src, output_dir / "tide_alas.csv")

        # Copier les scripts
        scripts = ["reformat.py", "comparaison.py", "dav.py", "currentprop.py", "avg_5m.py"]
        for s in scripts:
            shutil.copy(s, work_dir / s)

        # Ajuster chemins dans reformat.py et comparaison.py
        for script in ["reformat.py", "comparaison.py"]:
            script_path = work_dir / script
            text = script_path.read_text()
            text = text.replace(
                "Path('Orde_1 - CP_PRODUCT')",
                "Path('.') / 'Orde_1 - CP_PRODUCT'"
            )
            script_path.write_text(text)

        # -----------------------------
        # 3️⃣ Lancer reformat.py
        # -----------------------------
        env = os.environ.copy()
        env["ADCP_OUTPATH"] = str(output_dir)

        try:
            subprocess.run(
                 [sys.executable, "reformat.py"],
                cwd=str(work_dir),
                check=True,
                capture_output=True,
                text=True,
                env=env
            )
            st.success("Reformat terminé !")
        except subprocess.CalledProcessError as e:
            st.error(f"Erreur lors du reformat :\n{e.stderr}")
            st.stop()

        # -----------------------------
        # 4️⃣ Lancer comparaison.py
        # -----------------------------
        try:
            result = subprocess.run(
                [sys.executable, "comparaison.py"],
                cwd=str(work_dir),
                check=True,
                capture_output=True,
                text=True,
                env=env
            )
            st.success("Analyse comparative terminée !")
            st.text("STDOUT:\n" + result.stdout)
            st.text("STDERR:\n" + result.stderr)
        except subprocess.CalledProcessError as e:
            st.error("Erreur lors de l'analyse comparative !")
            st.text("STDOUT:\n" + e.stdout)
            st.text("STDERR:\n" + e.stderr)
            st.stop()

        # -----------------------------
        # 5️⃣ Navigation par onglets
        # -----------------------------
        tab1, tab2 = st.tabs(["Transects Individuels", "Analyse Comparative Globale"])

        # Onglet Transects Individuels
        with tab1:
            st.header("Transects Individuels")
            transect_img_folder = output_dir / "Transect_Image_Profile"
            transect_csv_folder = output_dir / "Transect_file"

            if transect_img_folder.exists():
                transect_imgs = sorted(transect_img_folder.glob("*.jpg"))
                for img_path in transect_imgs:
                    file_stem = img_path.stem
                    with st.expander(f"{file_stem}"):
                        st.subheader(f"Fichier : {file_stem}")
                        st.image(str(img_path), use_column_width=True)

                        # Optionnel : afficher CSV associé
                        csv_file = transect_csv_folder / f"{file_stem.replace('_avg5m','test')}.csv"
                        if csv_file.exists():
                            df = pd.read_csv(csv_file)
                            st.dataframe(df.head(10))
            else:
                st.warning("Aucune image individuelle générée.")

        # Onglet Analyse Comparative Globale
        with tab2:
            st.header("Analyse Comparative Globale")
            global_img_folder = output_dir / "Analyse_Globale/graphiques_evolution_par_zone"
            comparison_csv = output_dir / "Analyse_Globale/comparaison_vitesses_par_zone.csv"

            if global_img_folder.exists():
                global_imgs = sorted(global_img_folder.glob("*.png"))
                for img_path in global_imgs:
                    st.subheader(img_path.stem)
                    st.image(str(img_path), use_column_width=True)
            else:
                st.warning("Aucune image comparative globale générée.")

            if comparison_csv.exists():
                st.subheader("Tableau comparatif des vitesses par zone")
                df_comp = pd.read_csv(comparison_csv)
                st.dataframe(df_comp)

        # -----------------------------
        # 6️⃣ Télécharger ZIP complet
        # -----------------------------
        zip_path = work_dir / "resultats_ADCP.zip"
        with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zf:
            for file in output_dir.rglob("*"):
                zf.write(file, file.relative_to(work_dir))

        with open(zip_path, "rb") as f:
            st.download_button(
                label="Télécharger tous les résultats (CSV, SHP, images) en ZIP",
                data=f,
                file_name="resultats_ADCP.zip",
            )
