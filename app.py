import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from kmodes.kprototypes import KPrototypes
from fpdf import FPDF
import matplotlib.pyplot as plt
import seaborn as sns
import os

# --- KONSTANTA GLOBAL ---
PRIMARY_COLOR = "#2C2F7F"
ACCENT_COLOR = "#7AA02F"
BACKGROUND_COLOR = "#EAF0FA"
TEXT_COLOR = "#26272E"
HEADER_BACKGROUND_COLOR = ACCENT_COLOR
SIDEBAR_HIGHLIGHT_COLOR = "#4A5BAA"
ACTIVE_BUTTON_BG_COLOR = "#3F51B5"
ACTIVE_BUTTON_TEXT_COLOR = "#FFFFFF"
ACTIVE_BUTTON_BORDER_COLOR = "#FFD700"

ID_COLS = ["No", "Nama", "JK", "Kelas"]
NUMERIC_COLS = ["Rata Rata Nilai Akademik", "Kehadiran"]
CATEGORICAL_COLS = ["Ekstrakurikuler Komputer", "Ekstrakurikuler Pertanian",
                    "Ekstrakurikuler Menjahit", "Ekstrakurikuler Pramuka"]
ALL_FEATURES_FOR_CLUSTERING = NUMERIC_COLS + CATEGORICAL_COLS

# --- CUSTOM CSS & HEADER ---
custom_css = f"""
<style>
    .stApp {{
        background-color: {BACKGROUND_COLOR};
        color: {TEXT_COLOR};
        font-family: 'Segoe UI', 'Roboto', 'Helvetica Neue', Arial, sans-serif;
    }}
    .main .block-container {{
        padding-top: 7.5rem;
        padding-right: 4rem;
        padding-left: 4rem;
        padding-bottom: 3rem;
        max-width: 1200px;
        margin: auto;
    }}
    [data-testid="stVerticalBlock"] > div:not(:last-child),
    [data-testid="stHorizontalBlock"] > div:not(:last-child) {{
        margin-bottom: 0.5rem !important;
        padding-bottom: 0px !important;
    }}
    .stVerticalBlock, .stHorizontalBlock {{
        gap: 1rem !important;
    }}
    h1, h2, h3, h4, h5, h6 {{
        margin-top: 1.5rem !important;
        margin-bottom: 0.8rem !important;
        padding-top: 0rem !important;
        padding-bottom: 0rem !important;
        color: {PRIMARY_COLOR};
        font-weight: 600;
    }}
    h1 {{ font-size: 2.5em; }}
    h2 {{ font-size: 2em; }}
    h3 {{ font-size: 1.5em; }}
    .stApp > div > div:first-child > div:nth-child(2) [data-testid="stText"] {{
        margin-top: 0.5rem !important;
        margin-bottom: 1rem !important;
        padding-top: 0 !important;
        padding-bottom: 0 !important;
        font-size: 0.95em;
        color: #666666;
    }}
    .stApp > div > div:first-child > div:nth-child(3) h1:first-child,
    .stApp > div > div:first-child > div:nth-child(3) h2:first-child,
    .stApp > div > div:first-child > div:nth-child(3) h3:first-child
    {{
        margin-top: 1rem !important;
    }}
    .stApp > div > div:first-child > div:nth-child(3) [data-testid="stAlert"]:first-child {{
        margin-top: 1.2rem !important;
    }}
    [data-testid="stSidebar"] {{
        background-color: {PRIMARY_COLOR};
        color: #ffffff;
        padding-top: 2.5rem;
    }}
    [data-testid="stSidebar"] * {{
        color: #ffffff;
    }}
    [data-testid="stSidebar"] .stButton > button {{
        background-color: {PRIMARY_COLOR} !important;
        color: white !important;
        border: none !important;
        padding: 12px 25px !important;
        text-align: left !important;
        width: 100% !important;
        font-size: 17px !important;
        font-weight: 500 !important;
        margin: 0 !important;
        border-radius: 0 !important;
        transition: background-color 0.2s, color 0.2s, border-left 0.2s, box-shadow 0.2s;
        display: flex !important;
        justify-content: flex-start !important;
        align-items: center;
        gap: 10px;
    }}
    [data-testid="stSidebar"] .stButton > button:hover {{
        background-color: {SIDEBAR_HIGHLIGHT_COLOR} !important;
        color: #e0e0e0 !important;
    }}
    [data-testid="stSidebar"] [data-testid="stButton"] {{
        margin-bottom: 0px !important;
        padding: 0px !important;
    }}
    [data-testid="stSidebar"] [data-testid="stVerticalBlock"] > div {{
        margin-bottom: 0px !important;
    }}
    [data-testid="stSidebar"] .st-sidebar-button-active {{
        background-color: {ACTIVE_BUTTON_BG_COLOR} !important;
        color: {ACTIVE_BUTTON_TEXT_COLOR} !important;
        border-left: 6px solid {ACTIVE_BUTTON_BORDER_COLOR} !important;
        box-shadow: inset 4px 0 10px rgba(0,0,0,0.4) !important;
    }}
    [data-testid="stSidebar"] .st-sidebar-button-active > button {{
        background-color: {ACTIVE_BUTTON_BG_COLOR} !important;
        color: {ACTIVE_BUTTON_TEXT_COLOR} !important;
        font-weight: 700 !important;
    }}
    [data-testid="stSidebar"] .stButton > button:not(.st-sidebar-button-active) {{
        border-left: 6px solid transparent !important;
        box-shadow: none !important;
    }}
    .custom-header {{
        background-color: {HEADER_BACKGROUND_COLOR};
        padding: 25px 40px;
        color: white;
        display: flex;
        justify-content: space-between;
        align-items: center;
        border-radius: 0;
        box-shadow: 0 5px 15px rgba(0,0,0,0.25);
        position: sticky;
        top: 0;
        left: 0;
        width: 100%;
        z-index: 1000;
        margin: 0 !important;
    }}
    .custom-header h1 {{
        margin: 0 !important;
        font-size: 32px;
        font-weight: bold;
        color: white;
    }}
    .custom-header .kanan {{
        font-weight: 600;
        font-size: 19px;
        color: white;
        opacity: 0.9;
        text-align: right;
    }}
    @media (max-width: 768px) {{
        .custom-header {{
            flex-direction: column;
            align-items: flex-start;
            padding: 15px 20px;
            text-align: left;
        }}
        .custom-header h1 {{
            font-size: 24px;
            margin-bottom: 5px !important;
        }}
        .custom-header .kanan {{
            font-size: 14px;
            text-align: left;
        }}
        .main .block-container {{
            padding-top: 10rem;
            padding-right: 1rem;
            padding-left: 1rem;
        }}
    }}
    .stAlert {{
        border-radius: 10px;
        padding: 15px;
        margin-bottom: 20px !important;
        margin-top: 20px !important;
        font-size: 0.95em;
        line-height: 1.5;
    }}
    .stAlert.info {{
        background-color: #e3f2fd;
        color: #1976D2;
        border-left: 6px solid #2196F3;
    }}
    .stAlert.success {{
        background-color: #e8f5e9;
        color: #388E3C;
        border-left: 6px solid #4CAF50;
    }}
    .stAlert.warning {{
        background-color: #fffde7;
        color: #FFA000;
        border-left: 6px solid #FFC107;
    }}
    .stAlert.error {{
        background-color: #ffebee;
        color: #D32F2F;
        border-left: 6px solid #F44336;
    }}
    .stForm {{
        background-color: white;
        padding: 25px;
        border-radius: 12px;
        box-shadow: 0 4px 12px rgba(0,0,0,0.1);
        margin-top: 25px !important;
        margin-bottom: 25px !important;
        border: 1px solid #e0e0e0;
    }}
    .stDataFrame, .stTable {{
        border-radius: 10px;
        box-shadow: 0 2px 8px rgba(0,0,0,0.1);
        margin-top: 30px !important;
        margin-bottom: 30px !important;
        border: 1px solid #e0e0e0;
    }}
    .stTable table th {{
        background-color: #f5f5f5 !important;
        color: {PRIMARY_COLOR} !important;
        font-weight: bold;
    }}
    .stTable table td {{
        padding: 8px 12px !important;
    }}
    .stButton > button {{
        background-color: {ACCENT_COLOR};
        color: white;
        padding: 10px 25px;
        border-radius: 8px;
        border: none;
        transition: background-color 0.2s ease-in-out, transform 0.1s ease-in-out;
        margin-top: 15px !important;
        margin-bottom: 8px !important;
        font-weight: 600;
        box-shadow: 0 2px 5px rgba(0,0,0,0.2);
    }}
    .stButton > button:hover {{
        background-color: {PRIMARY_COLOR};
        color: white;
        transform: translateY(-2px);
        box-shadow: 0 4px 8px rgba(0,0,0,0.25);
    }}
    .stButton > button:active {{
        transform: translateY(0);
        box-shadow: 0 1px 3px rgba(0,0,0,0.2);
    }}
    .stTextInput > div > div > input,
    .stNumberInput > div > div > input,
    .stDateInput > div > div > input,
    .stTimeInput > div > div > input {{
        border-radius: 8px;
        border: 1px solid #D1D1D1;
        padding: 10px 15px;
        margin-bottom: 8px !important;
        margin-top: 8px !important;
        background-color: white;
        box-shadow: inset 0 1px 3px rgba(0,0,0,0.05);
    }}
    .stTextInput label, .stNumberInput label, .stSelectbox label, .stCheckbox label, .stRadio label {{
        margin-bottom: 5px !important;
        padding-bottom: 0px !important;
        font-size: 0.98em;
        font-weight: 500;
        color: {TEXT_COLOR};
    }}
    div[data-testid="stSelectbox"] > div:first-child {{
        width: 480px;
        min-width: 300px;
    }}
    div[data-testid="stSelectbox"] > div > div > div > div[role="button"] {{
        width: 100% !important;
        white-space: normal;
        overflow: hidden;
        text-overflow: ellipsis;
        display: flex;
        align-items: center;
        height: auto;
        box-sizing: border-box;
        padding-right: 35px;
    }}
    div[role="listbox"][aria-orientation="vertical"] {{
        width: 500px !important;
        max-width: 600px !important;
        min-width: 400px !important;
        overflow-x: hidden !important;
        overflow-y: auto !important;
        box-sizing: border-box;
        border-radius: 8px;
        border: 1px solid #D1D1D1;
        box-shadow: 0 4px 12px rgba(0,0,0,0.1);
        background-color: white;
    }}
    div[role="option"] {{
        white-space: normal !important;
        word-wrap: break-word !important;
        padding-right: 15px !important;
        padding-left: 15px !important;
        line-height: 1.4;
        min-height: 38px;
        display: flex;
        align-items: center;
    }}
    div[role="option"]:hover {{
        background-color: #e0e0e0;
        color: {PRIMARY_COLOR};
    }}
    ::-webkit-scrollbar {{
        width: 10px;
    }}
    ::-webkit-scrollbar-thumb {{
        background: {ACCENT_COLOR};
        border-radius: 5px;
    }}
    ::-webkit-scrollbar-track {{
        background: #e9e9e9;
    }}
    .stCheckbox label, .stRadio label {{
        display: flex;
        align-items: center;
        cursor: pointer;
        user-select: none;
    }}
    .stCheckbox {{
        margin-bottom: 10px !important;
        margin-top: 10px !important;
    }}
    .stExpander {{
        border: 1px solid #e0e0e0;
        border-radius: 10px;
        box-shadow: 0 2px 5px rgba(0,0,0,0.05);
        margin-bottom: 20px;
    }}
    .stExpander > div > div > p {{
        font-weight: 600;
        color: {PRIMARY_COLOR};
    }}
    div[data-testid="column"] {{
        gap: 2rem;
    }}
    .stApp > div > div:first-child > div:nth-child(3) > div:first-child {{
        margin-top: 0rem !important;
    }}
    .login-container {{
        display: flex;
        flex-direction: column;
        justify-content: center;
        align-items: center;
        height: 80vh;
        text-align: center;
    }}
    .login-card {{
        background-color: white;
        padding: 50px 70px;
        border-radius: 15px;
        box-shadow: 0 10px 20px rgba(0,0,0,0.1);
        border: 1px solid #e0e0e0;
        width: 100%;
        max-width: 600px;
        margin-top: 50px;
    }}
    .login-card h2 {{
        color: {PRIMARY_COLOR};
        font-size: 2.2em;
        margin-bottom: 2rem;
    }}
    .stButton > button {{
        background-color: {ACCENT_COLOR};
        color: white;
        padding: 10px 25px;
        border-radius: 8px;
        border: none;
        transition: background-color 0.2s ease-in-out, transform 0.1s ease-in-out;
        margin-top: 15px !important;
        margin-bottom: 8px !important;
        font-weight: 600;
        box-shadow: 0 2px 5px rgba(0,0,0,0.2);
    }}
</style>
"""

header_html = f"""
<div class="custom-header">
    <div><h1>PENGELOMPOKAN SISWA</h1></div>
    <div class="kanan">MADRASAH ALIYAH AL-HIKMAH</div>
</div>
"""

st.set_page_config(page_title="Klasterisasi K-Prototype Siswa", layout="wide", initial_sidebar_state="expanded")
st.markdown(custom_css, unsafe_allow_html=True)
st.markdown(header_html, unsafe_allow_html=True)

# --- FUNGSI PEMBANTU (dengan caching) ---

def generate_pdf_profil_siswa(nama, data_siswa_dict, klaster, cluster_desc_map):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", "B", 16)
    pdf.set_text_color(44, 47, 127)
    pdf.cell(0, 10, "PROFIL SISWA - HASIL KLASTERISASI", ln=True, align='C')
    pdf.ln(10)
    pdf.set_font("Arial", "", 10)
    pdf.set_text_color(0, 0, 0)
    keterangan_umum = (
        "Laporan ini menyajikan profil detail siswa berdasarkan hasil pengelompokan "
        "menggunakan Algoritma K-Prototype. Klasterisasi dilakukan berdasarkan "
        "nilai akademik, kehadiran, dan partisipasi ekstrakurikuler siswa. "
        "Informasi klaster ini dapat digunakan untuk memahami kebutuhan siswa dan "
        "merancang strategi pembinaan yang sesuai."
    )
    pdf.multi_cell(0, 5, keterangan_umum, align='J')
    pdf.ln(5)
    pdf.set_font("Arial", "B", 12)
    pdf.cell(0, 8, f"Nama Siswa: {nama}", ln=True)
    pdf.cell(0, 8, f"Klaster Hasil: {klaster}", ln=True)
    pdf.ln(3)
    klaster_desc = cluster_desc_map.get(klaster, "Deskripsi klaster tidak tersedia.")
    pdf.set_font("Arial", "I", 10)
    pdf.set_text_color(80, 80, 80)
    pdf.multi_cell(0, 5, f"Karakteristik Klaster {klaster}: {klaster_desc}", align='J')
    pdf.ln(5)
    pdf.set_font("Arial", "", 10)
    pdf.set_text_color(0, 0, 0)
    ekskul_diikuti = []
    ekskul_cols_full_names = ["Ekstrakurikuler Komputer", "Ekstrakurikuler Pertanian", "Ekstrakurikuler Menjahit", "Ekstrakurikuler Pramuka"]
    for col in ekskul_cols_full_names:
        val = data_siswa_dict.get(col)
        if val is not None and (val == 1 or str(val).strip() == '1'):
            ekskul_diikuti.append(col.replace("Ekstrakurikuler ", ""))

    display_data = {
        "Nomor Induk": data_siswa_dict.get("No", "-"),
        "Jenis Kelamin": data_siswa_dict.get("JK", "-"),
        "Kelas": data_siswa_dict.get("Kelas", "-"),
        "Rata-rata Nilai Akademik": f"{data_siswa_dict.get('Rata Rata Nilai Akademik', '-'):.2f}",
        "Persentase Kehadiran": f"{data_siswa_dict.get('Kehadiran', '-'):.2%}",
        "Ekstrakurikuler yang Diikuti": ", ".join(ekskul_diikuti) if ekskul_diikuti else "Tidak mengikuti ekstrakurikuler",
    }
    for key, val in display_data.items():
        pdf.cell(0, 7, f"{key}: {val}", ln=True)
    try:
        pdf_output = pdf.output(dest='S').encode('latin-1')
        return bytes(pdf_output)
    except Exception as e:
        st.error(f"Error saat mengonversi PDF: {e}. Coba pastikan tidak ada karakter aneh pada data.")
        return None

@st.cache_data(show_spinner="Sedang memproses dan menormalisasi data...")
def preprocess_data(df):
    df_processed = df.copy()
    df_processed.columns = [col.strip() for col in df_processed.columns]
    missing_cols = [col for col in NUMERIC_COLS + CATEGORICAL_COLS if col not in df_processed.columns]
    if missing_cols:
        st.error(f"Kolom-kolom berikut tidak ditemukan dalam data Anda: {', '.join(missing_cols)}. Harap periksa file Excel Anda dan pastikan nama kolom sudah benar.")
        return None, None
    df_clean_for_clustering = df_processed.drop(columns=ID_COLS, errors="ignore")
    for col in CATEGORICAL_COLS:
        df_clean_for_clustering[col] = df_clean_for_clustering[col].fillna(0).astype(str)
    for col in NUMERIC_COLS:
        if df_clean_for_clustering[col].isnull().any():
            mean_val = df_clean_for_clustering[col].mean()
            df_clean_for_clustering[col] = df_clean_for_clustering[col].fillna(mean_val)
            st.warning(f"Nilai kosong pada kolom '{col}' diisi dengan rata-rata: {mean_val:.2f}.")
    scaler = StandardScaler()
    df_clean_for_clustering[NUMERIC_COLS] = scaler.fit_transform(df_clean_for_clustering[NUMERIC_COLS])
    return df_clean_for_clustering, scaler

@st.cache_resource(show_spinner="Melakukan klasterisasi data...")
def run_kprototypes_clustering(df_preprocessed, n_clusters):
    df_for_clustering = df_preprocessed.copy()
    X_data = df_for_clustering[ALL_FEATURES_FOR_CLUSTERING]
    X = X_data.to_numpy()
    categorical_feature_indices = [X_data.columns.get_loc(c) for c in CATEGORICAL_COLS]
    try:
        kproto = KPrototypes(n_clusters=n_clusters, init='Huang', n_init=10, verbose=0, random_state=42, n_jobs=-1)
        clusters = kproto.fit_predict(X, categorical=categorical_feature_indices)
    except Exception as e:
        st.error(f"Terjadi kesalahan saat menjalankan K-Prototypes: {e}. Pastikan data Anda cukup bervariasi untuk jumlah klaster yang dipilih.")
        return None, None, None
    df_for_clustering["Klaster"] = clusters
    return df_for_clustering, kproto, categorical_feature_indices

@st.cache_data(show_spinner="Membuat deskripsi klaster...")
def generate_cluster_descriptions(df_clustered, n_clusters, numeric_cols, categorical_cols, df_original):
    cluster_characteristics_map = {}
    if df_original is None:
        return {}
    
    df_original_numeric = df_original[NUMERIC_COLS]
    for i in range(n_clusters):
        cluster_data = df_clustered[df_clustered["Klaster"] == i]
        avg_scaled_values = cluster_data[numeric_cols].mean()
        mode_values = cluster_data[categorical_cols].mode().iloc[0]
        desc = ""
        if avg_scaled_values["Rata Rata Nilai Akademik"] > 0.75:
            desc += "Siswa di klaster ini memiliki nilai akademik cenderung sangat tinggi. "
        elif avg_scaled_values["Rata Rata Nilai Akademik"] > 0.25:
            desc += "Siswa di klaster ini memiliki nilai akademik cenderung di atas rata-rata. "
        elif avg_scaled_values["Rata Rata Nilai Akademik"] < -0.75:
            desc += "Siswa di klaster ini memiliki nilai akademik cenderung sangat rendah. "
        elif avg_scaled_values["Rata Rata Nilai Akademik"] < -0.25:
            desc += "Siswa di klaster ini memiliki nilai akademik cenderung di bawah rata-rata. "
        else:
            desc += "Siswa di klaster ini memiliki nilai akademik cenderung rata-rata. "
        if avg_scaled_values["Kehadiran"] > 0.75:
            desc += "Tingkat kehadiran cenderung sangat tinggi. "
        elif avg_scaled_values["Kehadiran"] > 0.25:
            desc += "Tingkat kehadiran cenderung di atas rata-rata. "
        elif avg_scaled_values["Kehadiran"] < -0.75:
            desc += "Tingkat kehadiran cenderung sangat rendah. "
        elif avg_scaled_values["Kehadiran"] < -0.25:
            desc += "Tingkat kehadiran cenderung di bawah rata-rata. "
        else:
            desc += "Tingkat kehadiran cenderung rata-rata. "
        ekskul_aktif_modes = [col_name for col_name in categorical_cols if mode_values[col_name] == '1']
        if ekskul_aktif_modes:
            desc += f"Siswa di klaster ini aktif dalam ekstrakurikuler: {', '.join([c.replace('Ekstrakurikuler ', '') for c in ekskul_aktif_modes])}."
        else:
            desc += "Siswa di klaster ini kurang aktif dalam kegiatan ekstrakurikuler."
        cluster_characteristics_map[i] = desc
    return cluster_characteristics_map


# --- INISIALISASI SESSION STATE ---
if 'role' not in st.session_state:
    st.session_state.role = None
if 'df_original' not in st.session_state:
    st.session_state.df_original = None
if 'df_preprocessed_for_clustering' not in st.session_state:
    st.session_state.df_preprocessed_for_clustering = None
if 'df_clustered' not in st.session_state:
    st.session_state.df_clustered = None
if 'scaler' not in st.session_state:
    st.session_state.scaler = None
if 'kproto_model' not in st.session_state:
    st.session_state.kproto_model = None
if 'categorical_features_indices' not in st.session_state:
    st.session_state.categorical_features_indices = None
if 'n_clusters' not in st.session_state:
    st.session_state.n_clusters = 3
if 'cluster_characteristics_map' not in st.session_state:
    st.session_state.cluster_characteristics_map = {}
if 'current_menu' not in st.session_state:
    st.session_state.current_menu = None
if 'kepsek_current_menu' not in st.session_state:
    st.session_state.kepsek_current_menu = "Lihat Hasil Klasterisasi"


# --- FUNGSI HALAMAN UTAMA (UNTUK SETIAP PERAN) ---

def show_operator_tu_page():
    st.sidebar.title("MENU NAVIGASI")
    st.sidebar.markdown("---")
    
    menu_options = [
        "Unggah Data",
        "Praproses & Normalisasi Data",
        "Klasterisasi Data K-Prototypes",
        "Prediksi Klaster Siswa Baru",
        "Visualisasi & Profil Klaster",
        "Lihat Profil Siswa Individual"
    ]
    if 'current_menu' not in st.session_state or st.session_state.current_menu not in menu_options:
        st.session_state.current_menu = menu_options[0]

    for option in menu_options:
        icon_map = {
            "Unggah Data": "⬆",
            "Praproses & Normalisasi Data": "⚙",
            "Klasterisasi Data K-Prototypes": "📊",
            "Prediksi Klaster Siswa Baru": "🔮",
            "Visualisasi & Profil Klaster": "📈",
            "Lihat Profil Siswa Individual": "👤"
        }
        display_name = f"{icon_map.get(option, '')} {option}"
        button_key = f"nav_button_{option.replace(' ', '_').replace('&', 'and')}"

        if st.sidebar.button(display_name, key=button_key):
            st.session_state.current_menu = option
            st.rerun()

    # Logika untuk menandai tombol aktif
    js_highlight_active_button = f"""
    <script>
        function cleanButtonText(text) {{
            return (text || '').replace(/\\p{{Emoji}}/gu, '').trim();
        }}
        function highlightActiveSidebarButton() {{
            var currentMenu = '{st.session_state.current_menu}';
            var cleanCurrentMenuName = cleanButtonText(currentMenu);
            var sidebarButtonContainers = window.parent.document.querySelectorAll('[data-testid="stSidebar"] [data-testid="stButton"]');
            sidebarButtonContainers.forEach(function(container) {{
                var button = container.querySelector('button');
                if (button) {{
                    var buttonText = cleanButtonText(button.innerText || button.textContent);
                    container.classList.remove('st-sidebar-button-active');
                    if (buttonText === cleanCurrentMenuName) {{
                        container.classList.add('st-sidebar-button-active');
                    }}
                }}
            }});
        }}
        const observer = new MutationObserver((mutationsList, observer) => {{
            const sidebarChanged = mutationsList.some(mutation =>
                mutation.target.closest('[data-testid="stSidebar"]')
            );
            if (sidebarChanged) {{
                highlightActiveSidebarButton();
            }}
        }});
        observer.observe(window.parent.document.body, {{ childList: true, subtree: true }});
        highlightActiveSidebarButton();
    </script>
    """
    if hasattr(st, 'html'):
        st.html(js_highlight_active_button)
    else:
        st.markdown(js_highlight_active_button, unsafe_allow_html=True)
    
    st.sidebar.markdown("---")
    if st.sidebar.button("🚪 Keluar", key="logout_tu_sidebar"):
        st.session_state.clear()
        st.rerun()

    if st.session_state.current_menu == "Unggah Data":
        st.header("Unggah Data Siswa")
        st.markdown("""
        <div style='background-color:#e3f2fd; padding:15px; border-radius:10px; border-left: 5px solid #2196F3;'>
        Silakan unggah file Excel (.xlsx) yang berisi dataset siswa. Pastikan file Anda memiliki
        kolom-kolom berikut agar sistem dapat bekerja dengan baik:<br><br>
        <ul>
            <li><b>Kolom Identitas:</b> "No", "Nama", "JK", "Kelas"</li>
            <li><b>Kolom Numerik (untuk analisis):</b> "Rata Rata Nilai Akademik", "Kehadiran"</li>
            <li><b>Kolom Kategorikal (untuk analisis, nilai 0 atau 1):</b> "Ekstrakurikuler Komputer", "Ekstrakurikuler Pertanian", "Ekstrakurikuler Menjahit", "Ekstrakurikuler Pramuka"</li>
        </ul>
        Pastikan nama kolom sudah persis sama dan tidak ada kesalahan penulisan.
        </div>
        """, unsafe_allow_html=True)
        st.markdown("---")
        uploaded_file = st.file_uploader("Pilih File Excel Dataset", type=["xlsx"], help="Unggah file Excel Anda di sini. Hanya format .xlsx yang didukung.")
        if uploaded_file:
            try:
                df = pd.read_excel(uploaded_file, engine='openpyxl')
                st.session_state.df_original = df
                st.session_state.df_clustered = None
                st.success("Data berhasil diunggah! Anda dapat melanjutkan ke langkah praproses.")
                st.subheader("Preview Data yang Diunggah:")
                st.dataframe(df, use_container_width=True, height=300)
                st.markdown("<div style='margin-top: 20px;'></div>", unsafe_allow_html=True)
            except Exception as e:
                st.error(f"Terjadi kesalahan saat membaca file: {e}. Pastikan format file Excel benar dan tidak rusak.")

    elif st.session_state.current_menu == "Praproses & Normalisasi Data":
        st.header("Praproses Data & Normalisasi Z-score")
        if st.session_state.df_original is None or st.session_state.df_original.empty:
            st.warning("Silakan unggah data terlebih dahulu di menu 'Unggah Data'.")
        else:
            st.markdown("""
            <div style='background-color:#e3f2fd; padding:15px; border-radius:10px; border-left: 5px solid #2196F3;'>
            Pada tahap ini, data akan disiapkan untuk analisis klasterisasi. Proses yang dilakukan meliputi:
            <ul>
                <li><b>Pembersihan Data:</b> Menangani nilai-nilai yang hilang (missing values) pada kolom numerik (diisi dengan rata-rata).</li>
                <li><b>Konversi Tipe Data:</b> Memastikan kolom kategorikal memiliki tipe data yang sesuai untuk algoritma.</li>
                <li><b>Normalisasi Z-score:</b> Mengubah skala fitur numerik (nilai akademik & kehadiran) agar memiliki rata-rata nol dan deviasi standar satu, sehingga semua fitur memiliki bobot yang setara dalam perhitungan klasterisasi.</li>
            </ul>
            </div>
            """, unsafe_allow_html=True)
            st.markdown("---")
            if st.button("Jalankan Praproses & Normalisasi"):
                df_preprocessed, scaler = preprocess_data(st.session_state.df_original)
                if df_preprocessed is not None and scaler is not None:
                    st.session_state.df_preprocessed_for_clustering = df_preprocessed
                    st.session_state.scaler = scaler
                    st.success("Praproses dan Normalisasi berhasil dilakukan. Data siap untuk klasterisasi!")
                    st.subheader("Data Setelah Praproses dan Normalisasi:")
                    st.dataframe(st.session_state.df_preprocessed_for_clustering, use_container_width=True, height=300)
                    st.markdown("<div style='margin-top: 20px;'></div>", unsafe_allow_html=True)

    elif st.session_state.current_menu == "Klasterisasi Data K-Prototypes":
        st.header("Klasterisasi K-Prototypes")
        if st.session_state.df_preprocessed_for_clustering is None or st.session_state.df_preprocessed_for_clustering.empty:
            st.warning("Silakan lakukan praproses data terlebih dahulu di menu 'Praproses & Normalisasi Data'.")
        else:
            st.markdown("""
            <div style='background-color:#e3f2fd; padding:15px; border-radius:10px; border-left: 5px solid #2196F3;'>
            Pada tahap ini, Anda akan menjalankan algoritma K-Prototypes untuk mengelompokkan siswa.
            <br><br>
            Pilih <b>Jumlah Klaster (K)</b> yang Anda inginkan (antara 2 hingga 6). Algoritma ini akan
            mengelompokkan siswa berdasarkan kombinasi fitur numerik (nilai akademik, kehadiran) dan
            fitur kategorikal (ekstrakurikuler) yang telah disiapkan sebelumnya.
            </div>
            """, unsafe_allow_html=True)
            st.markdown("---")
            k = st.slider("Pilih Jumlah Klaster (K)", 2, 6, value=st.session_state.n_clusters,
                            help="Pilih berapa banyak kelompok siswa yang ingin Anda bentuk.")
            if st.button("Jalankan Klasterisasi"):
                df_clustered, kproto_model, categorical_features_indices = run_kprototypes_clustering(
                    st.session_state.df_preprocessed_for_clustering, k
                )
                if df_clustered is not None:
                    df_final = st.session_state.df_original.copy()
                    df_final['Klaster'] = df_clustered['Klaster']
                    
                    # --- PERBAIKAN: Simpan langsung ke session state ---
                    st.session_state.df_clustered = df_final
                    st.session_state.kproto_model = kproto_model
                    st.session_state.categorical_features_indices = categorical_features_indices
                    st.session_state.n_clusters = k
                    st.session_state.cluster_characteristics_map = generate_cluster_descriptions(
                        df_clustered, k, NUMERIC_COLS, CATEGORICAL_COLS, st.session_state.df_original
                    )

                    st.success(f"Klasterisasi selesai dengan {k} klaster! Hasil pengelompokan siswa telah tersedia.")
                    st.markdown("---")
                    st.subheader("Data Hasil Klasterisasi (Disertai Data Asli):")
                    st.dataframe(df_final, use_container_width=True, height=300)
                    st.markdown("<div style='margin-top: 30px;'></div>", unsafe_allow_html=True)
                    st.subheader("Ringkasan Klaster: Jumlah Siswa per Kelompok")
                    jumlah_per_klaster = df_final["Klaster"].value_counts().sort_index().reset_index()
                    jumlah_per_klaster.columns = ["Klaster", "Jumlah Siswa"]
                    st.table(jumlah_per_klaster)
                    st.markdown("<div style='margin-top: 30px;'></div>", unsafe_allow_html=True)
                    st.subheader(f"Karakteristik Umum Klaster ({st.session_state.n_clusters} Klaster):")
                    st.write("Berikut adalah deskripsi singkat untuk setiap klaster yang terbentuk:")
                    for cluster_id, desc in st.session_state.cluster_characteristics_map.items():
                        with st.expander(f"Klaster {cluster_id}"):
                            st.markdown(desc)
    # ... (sisanya tidak berubah) ...

def show_kepala_sekolah_page():
    st.sidebar.title("MENU NAVIGASI")
    st.sidebar.markdown("---")
    
    kepsek_menu_options = [
        "Lihat Hasil Klasterisasi",
        "Visualisasi & Profil Klaster",
        "Lihat Profil Siswa Individual"
    ]
    if 'kepsek_current_menu' not in st.session_state:
        st.session_state.kepsek_current_menu = kepsek_menu_options[0]

    for option in kepsek_menu_options:
        icon_map = {
            "Lihat Hasil Klasterisasi": "📋",
            "Visualisasi & Profil Klaster": "📈",
            "Lihat Profil Siswa Individual": "👤"
        }
        display_name = f"{icon_map.get(option, '')} {option}"
        button_key = f"kepsek_nav_button_{option.replace(' ', '_').replace('&', 'and')}"

        if st.sidebar.button(display_name, key=button_key):
            st.session_state.kepsek_current_menu = option
            st.rerun()

    js_highlight_active_button = f"""
    <script>
        function cleanButtonText(text) {{
            return (text || '').replace(/\\p{{Emoji}}/gu, '').trim();
        }}
        function highlightActiveSidebarButton() {{
            var currentMenu = '{st.session_state.kepsek_current_menu}';
            var cleanCurrentMenuName = cleanButtonText(currentMenu);
            var sidebarButtonContainers = window.parent.document.querySelectorAll('[data-testid="stSidebar"] [data-testid="stButton"]');
            sidebarButtonContainers.forEach(function(container) {{
                var button = container.querySelector('button');
                if (button) {{
                    var buttonText = cleanButtonText(button.innerText || button.textContent);
                    container.classList.remove('st-sidebar-button-active');
                    if (buttonText === cleanCurrentMenuName) {{
                        container.classList.add('st-sidebar-button-active');
                    }}
                }}
            }});
        }}
        const observer = new MutationObserver((mutationsList, observer) => {{
            const sidebarChanged = mutationsList.some(mutation =>
                mutation.target.closest('[data-testid="stSidebar"]')
            );
            if (sidebarChanged) {{
                highlightActiveSidebarButton();
            }}
        }});
        observer.observe(window.parent.document.body, {{ childList: true, subtree: true }});
        highlightActiveSidebarButton();
    </script>
    """
    if hasattr(st, 'html'):
        st.html(js_highlight_active_button)
    else:
        st.markdown(js_highlight_active_button, unsafe_allow_html=True)
    
    st.sidebar.markdown("---")
    if st.sidebar.button("🚪 Keluar", key="logout_kepsek_sidebar"):
        st.session_state.clear()
        st.rerun()
    
    st.title("👨‍💼 Dasbor Kepala Sekolah")
    
    # --- PERBAIKAN: Membaca dari session_state, bukan dari file ---
    if st.session_state.df_clustered is None:
        st.warning(f"Data hasil klasterisasi belum tersedia. Mohon minta Operator TU untuk memproses data terlebih dahulu.")
        return

    df_kepsek = st.session_state.df_clustered
    # ... (sisanya sama) ...
