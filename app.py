import streamlit as st
from pathlib import Path
import subprocess
import zipfile
import shutil
import os
import pandas as pd
import sys
st.set_page_config(page_title="Pipeline ADCP", layout="wide")
st.title("Pipeline ADCP - Reformat e Análise Comparativa (Velocidade & Backscatter)") 

# -----------------------------
# 1️⃣ Upload dos arquivos RAW (inalterado)
# -----------------------------
raw_files = st.file_uploader(
    "Envie seus arquivos RAW (_ASC.txt)", type="txt", accept_multiple_files=True
)

if st.button("Iniciar todo o processamento (Velocidade & Backscatter)"):

    if not raw_files:
        st.error("É necessário enviar pelo menos os arquivos RAW.")
        st.stop()
    else:
        st.info("Processamento em andamento, isso pode levar alguns minutos...")

        # -----------------------------
        # 2️⃣ Preparação da pasta de trabalho (inalterado)
        # -----------------------------
        work_dir = Path("streamlit_temp").resolve()
        if work_dir.exists():
            shutil.rmtree(work_dir)
        work_dir.mkdir(parents=True)

        raw_folder = work_dir / "Orde_0 - CP_RAW/PLAYBACK DATA"
        raw_folder.mkdir(parents=True, exist_ok=True)
        for f in raw_files:
            (raw_folder / f.name).write_bytes(f.read())

        # Arquivos fixos
        fixed_files_dir = Path("fixed_files")
        metadata_src = fixed_files_dir / "metadata.txt"
        tide_src = fixed_files_dir / "tide_alas.csv"
        output_dir = work_dir / "Orde_1 - CP_PRODUCT"
        output_dir.mkdir(parents=True, exist_ok=True)

        if metadata_src.exists():
            shutil.copy(metadata_src, output_dir / "metadata.txt")
        if tide_src.exists():
            shutil.copy(tide_src, output_dir / "tide_alas.csv")

        # Copiar scripts necessários
        scripts = ["reformat.py", "comparaison.py", "dav.py", "currentprop.py", "avg_5m.py"]
        for s in scripts:
            if Path(s).exists():
                shutil.copy(s, work_dir / s)
            else:
                st.error(f"O script necessário '{s}' não foi encontrado. Certifique-se de que ele esteja na mesma pasta que o aplicativo Streamlit.")
                st.stop()

        # Ajustar caminhos em reformat.py e comparaison.py
        for script in ["reformat.py", "comparaison.py"]:
            script_path = work_dir / script
            if script_path.exists():
                text = script_path.read_text()
                text = text.replace(
                    "Path('Orde_1 - CP_PRODUCT')",
                    "Path('.') / 'Orde_1 - CP_PRODUCT'"
                )
                script_path.write_text(text)

        # -----------------------------
        # 3️⃣ Execução do reformat.py (inalterado)
        # -----------------------------
        env = os.environ.copy()
        env["ADCP_OUTPATH"] = str(output_dir)
        progress_bar = st.progress(0, text="Etapa 1/2 : Reformatando os dados...")

        try:
            subprocess.run(
                [python_executable, "reformat.py"],
                cwd=str(work_dir), check=True, capture_output=True, text=True, env=env
            )
            st.success("Reformatação dos dados concluída!")
            progress_bar.progress(50, text="Etapa 2/2 : Iniciando as análises...")
        except subprocess.CalledProcessError as e:
            st.error(f"Erro crítico durante a reformatação:\n{e.stderr}")
            st.code(e.stdout)
            st.stop()

        # -----------------------------
        # 4️⃣ Execução do comparaison.py (inalterado)
        # -----------------------------
        try:
            result = subprocess.run(
                [python_executable,"comparaison.py"],
                cwd=str(work_dir), check=True, capture_output=True, text=True, env=env
            )
            st.success("Análises de Velocidade & Backscatter concluídas!")
            progress_bar.progress(100, text="Processamento concluído!")
            with st.expander("Ver logs de saída do script de análise"):
                 st.text("STDOUT:\n" + result.stdout)
                 st.text("STDERR:\n" + result.stderr)
        except subprocess.CalledProcessError as e:
            st.error("Erro crítico durante a análise comparativa!")
            st.text("STDOUT:\n" + e.stdout)
            st.text("STDERR:\n" + e.stderr)
            st.stop()

        # =========================================================================
        # ====> MODIFICAÇÃO MAIOR : Nova estrutura de exibição dos resultados <====
        # =========================================================================
        
        st.header("Visualização dos Resultados")
        
        tab1, tab2 = st.tabs(["📊 Transectos Individuais", "📈 Análise Comparativa Global"])

        # Aba 1 : Transectos Individuais
        with tab1:
            st.subheader("Perfis de Velocidade e Backscatter para cada transecto")

            # --- Caminhos para as pastas de resultados ---
            img_folder_vel = output_dir / "Transect_Image_Profile"
            img_folder_bs = output_dir / "Transect_Image_Profile_BS"

            all_vel_imgs = sorted(img_folder_vel.glob("*.jpg")) if img_folder_vel.exists() else []
            all_bs_imgs = sorted(img_folder_bs.glob("*.jpg")) if img_folder_bs.exists() else []
            
            # Criar um dicionário para associar imagens pelo nome base do arquivo
            image_pairs = {}
            for img_path in all_vel_imgs:
                base_name = img_path.stem.replace('_avg5m', '')
                image_pairs.setdefault(base_name, {})['vel'] = img_path
            
            for img_path in all_bs_imgs:
                base_name = img_path.stem.replace('_backscatter', '')
                image_pairs.setdefault(base_name, {})['bs'] = img_path

            if not image_pairs:
                st.warning("Nenhuma imagem de transecto individual foi gerada.")
            else:
                for base_name, paths in sorted(image_pairs.items()):
                    with st.expander(f"Transecto : {base_name}"):
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            st.markdown("#### Perfil de Velocidade")
                            if 'vel' in paths:
                                st.image(str(paths['vel']), use_column_width=True, caption="Vetores de velocidade da corrente.")
                            else:
                                st.info("Nenhuma imagem de velocidade para este transecto.")
                        
                        with col2:
                            st.markdown("#### Perfil de Backscatter")
                            if 'bs' in paths:
                                st.image(str(paths['bs']), use_column_width=True, caption="Intensidade do backscatter acústico (dB).")
                            else:
                                st.info("Nenhuma imagem de backscatter para este transecto.")

        # Aba 2 : Análise Comparativa Global
        with tab2:
            st.subheader("Evolução temporal das médias por zona")
            
            col1, col2 = st.columns(2)

            # --- Coluna da ESQUERDA : VELOCIDADE ---
            with col1:
                st.markdown("### 💨 Análise da Velocidade")
                
                # Caminhos para velocidade
                global_img_folder_vel = output_dir / "Analyse_Globale/graphiques_evolution_par_zone"
                comparison_csv_vel = output_dir / "Analyse_Globale/comparaison_vitesses_par_zone.csv"

                # Exibição da tabela comparativa de velocidade
                if comparison_csv_vel.exists():
                    st.markdown("##### Tabela comparativa das velocidades médias (m/s)")
                    df_comp_vel = pd.read_csv(comparison_csv_vel)
                    st.dataframe(df_comp_vel)
                else:
                    st.warning("A tabela comparativa de velocidades não foi encontrada.")

                # Exibição dos gráficos de evolução da velocidade
                if global_img_folder_vel.exists():
                    st.markdown("##### Gráficos de evolução por zona")
                    global_imgs_vel = sorted(global_img_folder_vel.glob("*.png"))
                    for img_path in global_imgs_vel:
                        st.image(str(img_path), use_column_width=True)
                else:
                    st.warning("Nenhum gráfico comparativo para velocidade foi gerado.")
            
            # --- Coluna da DIREITA : BACKSCATTER ---
            with col2:
                st.markdown("### 🌊 Análise do Backscatter")

                # Novos caminhos para o backscatter
                global_img_folder_bs = output_dir / "Analyse_Globale_BS/graphiques_evolution_par_zone_bs"
                comparison_csv_bs = output_dir / "Analyse_Globale_BS/comparaison_bs_par_zone.csv"
                
                # Exibição da tabela comparativa de backscatter
                if comparison_csv_bs.exists():
                    st.markdown("##### Tabela comparativa do backscatter médio (dB)")
                    df_comp_bs = pd.read_csv(comparison_csv_bs)
                    st.dataframe(df_comp_bs)
                else:
                    st.warning("A tabela comparativa de backscatter não foi encontrada.")
                
                # Exibição dos gráficos de evolução do backscatter
                if global_img_folder_bs.exists():
                    st.markdown("##### Gráficos de evolução por zona")
                    global_imgs_bs = sorted(global_img_folder_bs.glob("*.png"))
                    for img_path in global_imgs_bs:
                        st.image(str(img_path), use_column_width=True)
                else:
                    st.warning("Nenhum gráfico comparativo para backscatter foi gerado.")

        # -----------------------------
        # 6️⃣ Download do ZIP completo (inalterado e funcional)
        # -----------------------------
        st.header("Download dos resultados completos")
        zip_path = work_dir / "resultados_ADCP_completos.zip"
        with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zf:
            # Compactar a pasta de saída que contém TODOS os resultados (velocidade e bs)
            for file in output_dir.rglob("*"):
                zf.write(file, file.relative_to(work_dir))

        with open(zip_path, "rb") as f:
            st.download_button(
                label="📥 Baixar todos os resultados (CSV, SHP, imagens) em ZIP",
                data=f,
                file_name="resultados_ADCP_completos.zip",
                mime="application/zip"
            )
