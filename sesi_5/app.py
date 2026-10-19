"""
app.py
======
Dasbor Web Interaktif menggunakan Streamlit.
Menerapkan prinsip visualisasi data Shneiderman:
- Overview First: Grafik ringkasan utama di atas
- Zoom & Filter: Sidebar filter kategori dan rentang tanggal
- Details on Demand: Tabel detail di bagian bawah

Dataset: Simulasi data penjualan e-commerce (dummy)

Jalankan: streamlit run sesi_5/app.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta

# ==============================================================================
# KONFIGURASI HALAMAN
# ==============================================================================

st.set_page_config(
    page_title="E-Commerce Analytics Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Injeksi CSS kustom untuk tampilan premium
st.markdown("""
<style>
    /* Font dan latar utama */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&display=swap');
    html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
    
    /* Warna tema */
    :root {
        --primary: #6C63FF;
        --secondary: #FF6584;
        --accent: #43E6FC;
        --bg-card: #1E2130;
        --text-muted: #8892A4;
    }
    
    /* Kartu metrik */
    div[data-testid="metric-container"] {
        background: linear-gradient(135deg, #1E2130, #252A3C);
        border: 1px solid rgba(108, 99, 255, 0.3);
        border-radius: 12px;
        padding: 16px;
        box-shadow: 0 4px 20px rgba(0,0,0,0.2);
    }
    
    /* Header sidebar */
    .sidebar-header {
        background: linear-gradient(135deg, #6C63FF, #FF6584);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 1.3rem;
        font-weight: 700;
        margin-bottom: 1rem;
    }
    
    /* Section header */
    .section-header {
        border-left: 4px solid #6C63FF;
        padding-left: 12px;
        margin: 1.5rem 0 1rem 0;
        font-size: 1.1rem;
        font-weight: 600;
        color: #E0E6F0;
    }
    
    /* Tabel dataframe */
    div[data-testid="stDataFrame"] {
        border: 1px solid rgba(108, 99, 255, 0.2);
        border-radius: 8px;
    }
    
    /* Hide Streamlit default footer */
    footer {visibility: hidden;}
    
    /* Main container padding */
    .main .block-container { padding-top: 2rem; }
</style>
""", unsafe_allow_html=True)


# ==============================================================================
# 1. GENERATE DATA DUMMY
# ==============================================================================

@st.cache_data
def generate_sales_data(n_records: int = 2000, seed: int = 42) -> pd.DataFrame:
    """
    Membuat dataset penjualan e-commerce dummy yang realistis.
    Di-cache agar tidak di-regenerate setiap kali filter berubah.
    """
    np.random.seed(seed)

    categories    = ["Elektronik", "Fashion", "Makanan & Minuman", "Kesehatan", "Olahraga", "Buku"]
    regions       = ["Jawa", "Sumatera", "Kalimantan", "Sulawesi", "Papua & Maluku"]
    channels      = ["Marketplace", "Website", "Mobile App", "Social Commerce"]
    payment_types = ["Transfer Bank", "QRIS", "Kartu Kredit", "COD", "Dompet Digital"]

    # Buat tanggal dalam 12 bulan terakhir
    end_date   = datetime.now()
    start_date = end_date - timedelta(days=365)
    dates = [
        start_date + timedelta(days=np.random.randint(0, 365))
        for _ in range(n_records)
    ]

    # Simulasikan pola harga per kategori
    base_prices = {
        "Elektronik": (500_000, 5_000_000),
        "Fashion": (100_000, 1_000_000),
        "Makanan & Minuman": (20_000, 300_000),
        "Kesehatan": (50_000, 800_000),
        "Olahraga": (150_000, 2_000_000),
        "Buku": (30_000, 200_000),
    }

    cat_list = np.random.choice(categories, n_records)
    prices   = np.array([
        np.random.randint(*base_prices[c]) for c in cat_list
    ])
    qty      = np.random.randint(1, 10, n_records)
    revenue  = prices * qty
    rating   = np.round(np.random.uniform(3.0, 5.0, n_records), 1)

    df = pd.DataFrame({
        "tanggal":        pd.to_datetime(dates),
        "order_id":       [f"ORD-{i+10001}" for i in range(n_records)],
        "kategori":       cat_list,
        "region":         np.random.choice(regions, n_records),
        "channel":        np.random.choice(channels, n_records),
        "pembayaran":     np.random.choice(payment_types, n_records),
        "harga_satuan":   prices,
        "kuantitas":      qty,
        "total_revenue":  revenue,
        "rating":         rating,
        "diskon_pct":     np.random.choice([0, 5, 10, 15, 20, 25], n_records),
    })

    df["bulan"]     = df["tanggal"].dt.to_period("M").astype(str)
    df["hari_nama"] = df["tanggal"].dt.strftime("%A")

    return df


# ==============================================================================
# 2. SIDEBAR FILTER (Zoom & Filter — Prinsip Shneiderman)
# ==============================================================================

def render_sidebar(df: pd.DataFrame) -> pd.DataFrame:
    """
    Menampilkan sidebar dengan filter interaktif dan mengembalikan data yang sudah difilter.
    Implementasi prinsip 'Zoom and Filter' dari Shneiderman.
    """
    with st.sidebar:
        st.markdown('<p class="sidebar-header">🔍 Filter & Zoom</p>', unsafe_allow_html=True)
        st.caption("Sesuaikan filter untuk mempersempit tampilan data.")

        st.markdown("---")

        # Filter 1: Rentang tanggal
        st.markdown("**📅 Rentang Tanggal**")
        min_date = df["tanggal"].min().date()
        max_date = df["tanggal"].max().date()
        date_range = st.date_input(
            "Pilih periode:",
            value=(min_date, max_date),
            min_value=min_date,
            max_value=max_date,
            label_visibility="collapsed",
        )

        # Filter 2: Kategori produk
        st.markdown("**🏷️ Kategori Produk**")
        all_categories = sorted(df["kategori"].unique())
        selected_cats = st.multiselect(
            "Pilih kategori:",
            options=all_categories,
            default=all_categories,
            label_visibility="collapsed",
        )

        # Filter 3: Region
        st.markdown("**🗺️ Region**")
        all_regions = sorted(df["region"].unique())
        selected_regions = st.multiselect(
            "Pilih region:",
            options=all_regions,
            default=all_regions,
            label_visibility="collapsed",
        )

        # Filter 4: Channel penjualan
        st.markdown("**📱 Channel Penjualan**")
        all_channels = sorted(df["channel"].unique())
        selected_channels = st.multiselect(
            "Pilih channel:",
            options=all_channels,
            default=all_channels,
            label_visibility="collapsed",
        )

        # Filter 5: Revenue minimum
        st.markdown("**💰 Revenue Minimum per Order**")
        min_revenue, max_revenue = int(df["total_revenue"].min()), int(df["total_revenue"].max())
        revenue_threshold = st.slider(
            "Min revenue (Rp):",
            min_value=min_revenue,
            max_value=max_revenue,
            value=min_revenue,
            step=50_000,
            format="Rp %d",
            label_visibility="collapsed",
        )

        st.markdown("---")
        st.caption(f"Data periode: {min_date} — {max_date}")

    # Terapkan semua filter
    if len(date_range) == 2:
        start, end = date_range
        df = df[(df["tanggal"].dt.date >= start) & (df["tanggal"].dt.date <= end)]

    if selected_cats:
        df = df[df["kategori"].isin(selected_cats)]
    if selected_regions:
        df = df[df["region"].isin(selected_regions)]
    if selected_channels:
        df = df[df["channel"].isin(selected_channels)]

    df = df[df["total_revenue"] >= revenue_threshold]

    return df


# ==============================================================================
# 3. OVERVIEW METRICS (Overview First — Prinsip Shneiderman)
# ==============================================================================

def render_kpi_cards(df: pd.DataFrame) -> None:
    """
    Menampilkan KPI cards sebagai gambaran ringkasan cepat (Overview).
    """
    total_revenue = df["total_revenue"].sum()
    total_orders  = len(df)
    avg_order     = df["total_revenue"].mean() if total_orders > 0 else 0
    avg_rating    = df["rating"].mean() if total_orders > 0 else 0

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("💰 Total Revenue", f"Rp {total_revenue:,.0f}", help="Total pendapatan dari semua transaksi yang difilter")
    col2.metric("🛒 Total Order",   f"{total_orders:,}", help="Jumlah order setelah difilter")
    col3.metric("📦 Avg Order Value", f"Rp {avg_order:,.0f}", help="Rata-rata nilai per order")
    col4.metric("⭐ Avg Rating",    f"{avg_rating:.2f} / 5.0", help="Rata-rata rating produk")


# ==============================================================================
# 4. GRAFIK OVERVIEW (tanpa chartjunk — Prinsip Shneiderman)
# ==============================================================================

def render_overview_charts(df: pd.DataFrame) -> None:
    """
    Menampilkan grafik ringkasan utama. Desain bersih tanpa elemen berlebihan (no chartjunk).
    """
    st.markdown('<div class="section-header">📈 Overview Penjualan</div>', unsafe_allow_html=True)

    col1, col2 = st.columns([3, 2])

    # Grafik 1: Tren revenue bulanan (line chart bersih)
    with col1:
        monthly = (df.groupby("bulan")["total_revenue"].sum().reset_index()
                     .sort_values("bulan"))
        fig_trend = px.area(
            monthly, x="bulan", y="total_revenue",
            title="Tren Revenue Bulanan",
            labels={"bulan": "Bulan", "total_revenue": "Revenue (Rp)"},
            color_discrete_sequence=["#6C63FF"],
        )
        fig_trend.update_layout(
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
            font_color="#C0C8D8",
            title_font_size=14,
            showlegend=False,
            margin=dict(t=40, b=20, l=20, r=20),
            xaxis=dict(showgrid=False, tickangle=-45, tickfont_size=9),
            yaxis=dict(showgrid=True, gridcolor="rgba(255,255,255,0.08)"),
        )
        fig_trend.update_traces(line_color="#6C63FF", fillcolor="rgba(108,99,255,0.15)")
        st.plotly_chart(fig_trend, use_container_width=True)

    # Grafik 2: Revenue per kategori (horizontal bar bersih)
    with col2:
        cat_revenue = (df.groupby("kategori")["total_revenue"].sum()
                         .sort_values(ascending=True).reset_index())
        fig_cat = px.bar(
            cat_revenue, x="total_revenue", y="kategori",
            orientation="h",
            title="Revenue per Kategori",
            labels={"total_revenue": "Revenue (Rp)", "kategori": ""},
            color="total_revenue",
            color_continuous_scale=["#6C63FF", "#FF6584", "#43E6FC"],
        )
        fig_cat.update_layout(
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
            font_color="#C0C8D8",
            title_font_size=14,
            showlegend=False,
            coloraxis_showscale=False,
            margin=dict(t=40, b=20, l=20, r=20),
            xaxis=dict(showgrid=True, gridcolor="rgba(255,255,255,0.08)"),
            yaxis=dict(showgrid=False),
        )
        st.plotly_chart(fig_cat, use_container_width=True)

    # Row 2: Channel & Region
    col3, col4 = st.columns(2)

    with col3:
        channel_data = df.groupby("channel")["total_revenue"].sum().reset_index()
        fig_channel = px.pie(
            channel_data, names="channel", values="total_revenue",
            title="Distribusi Revenue per Channel",
            color_discrete_sequence=["#6C63FF", "#FF6584", "#43E6FC", "#FFE566"],
            hole=0.45,
        )
        fig_channel.update_layout(
            plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
            font_color="#C0C8D8", title_font_size=14,
            legend=dict(orientation="h", yanchor="bottom", y=-0.25),
            margin=dict(t=40, b=50, l=10, r=10),
        )
        fig_channel.update_traces(textposition="inside", textinfo="percent+label")
        st.plotly_chart(fig_channel, use_container_width=True)

    with col4:
        region_data = df.groupby("region")["total_revenue"].sum().sort_values(ascending=False).reset_index()
        fig_region = px.bar(
            region_data, x="region", y="total_revenue",
            title="Revenue per Region",
            labels={"region": "Region", "total_revenue": "Revenue (Rp)"},
            color="region",
            color_discrete_sequence=["#6C63FF", "#FF6584", "#43E6FC", "#FFE566", "#68D8A4"],
        )
        fig_region.update_layout(
            plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
            font_color="#C0C8D8", title_font_size=14, showlegend=False,
            margin=dict(t=40, b=20, l=20, r=20),
            xaxis=dict(showgrid=False),
            yaxis=dict(showgrid=True, gridcolor="rgba(255,255,255,0.08)"),
        )
        st.plotly_chart(fig_region, use_container_width=True)


# ==============================================================================
# 5. DETAILS ON DEMAND (Prinsip Shneiderman)
# ==============================================================================

def render_details_table(df: pd.DataFrame) -> None:
    """
    Menampilkan tabel detail data di bagian bawah (Details on Demand).
    """
    st.markdown('<div class="section-header">📋 Detail Data Transaksi</div>', unsafe_allow_html=True)

    # Kontrol tampilan tabel
    col1, col2, col3 = st.columns([2, 2, 1])
    with col1:
        sort_col = st.selectbox("Urutkan berdasarkan:", ["total_revenue", "tanggal", "rating", "kuantitas"])
    with col2:
        sort_dir = st.radio("Urutan:", ["Terbesar/Terbaru", "Terkecil/Terlama"], horizontal=True)
    with col3:
        n_rows = st.selectbox("Tampilkan:", [25, 50, 100, 200])

    ascending = sort_dir == "Terkecil/Terlama"
    display_df = (df.sort_values(by=sort_col, ascending=ascending)
                    .head(n_rows)
                    [["tanggal", "order_id", "kategori", "region", "channel",
                      "pembayaran", "harga_satuan", "kuantitas", "total_revenue",
                      "rating", "diskon_pct"]])

    display_df = display_df.rename(columns={
        "tanggal": "Tanggal", "order_id": "Order ID", "kategori": "Kategori",
        "region": "Region", "channel": "Channel", "pembayaran": "Pembayaran",
        "harga_satuan": "Harga Satuan (Rp)", "kuantitas": "Qty",
        "total_revenue": "Revenue (Rp)", "rating": "Rating ⭐", "diskon_pct": "Diskon (%)",
    })

    st.dataframe(
        display_df,
        use_container_width=True,
        hide_index=True,
        column_config={
            "Harga Satuan (Rp)": st.column_config.NumberColumn(format="Rp %d"),
            "Revenue (Rp)": st.column_config.NumberColumn(format="Rp %d"),
            "Rating ⭐": st.column_config.NumberColumn(format="%.1f ⭐"),
            "Diskon (%)": st.column_config.NumberColumn(format="%d%%"),
        }
    )

    st.caption(f"Menampilkan {min(n_rows, len(df))} dari {len(df):,} transaksi yang difilter.")

    # Tombol download
    csv_data = df.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="⬇️ Download Data (CSV)",
        data=csv_data,
        file_name=f"sales_data_filtered_{datetime.now().strftime('%Y%m%d')}.csv",
        mime="text/csv",
    )


# ==============================================================================
# 6. MAIN APP
# ==============================================================================

def main():
    # Header utama
    st.markdown("""
        <div style='text-align: center; padding: 1rem 0 0.5rem 0;'>
            <h1 style='font-size: 2rem; font-weight: 700; 
                       background: linear-gradient(135deg, #6C63FF, #FF6584, #43E6FC);
                       -webkit-background-clip: text; -webkit-text-fill-color: transparent;'>
                📊 E-Commerce Analytics Dashboard
            </h1>
            <p style='color: #8892A4; font-size: 0.95rem; margin-top: -0.5rem;'>
                Dasbor interaktif berbasis prinsip Shneiderman: Overview First → Zoom & Filter → Details on Demand
            </p>
        </div>
        <hr style='border-color: rgba(108,99,255,0.3); margin-bottom: 1.5rem;'>
    """, unsafe_allow_html=True)

    # Load data
    df_raw = generate_sales_data()
    total_raw = len(df_raw)

    # PRINSIP 2: Zoom & Filter (sidebar)
    df_filtered = render_sidebar(df_raw)

    # Status filter
    n_filtered = len(df_filtered)
    st.info(f"📊 Menampilkan **{n_filtered:,}** dari **{total_raw:,}** total transaksi setelah filter diterapkan.")

    if df_filtered.empty:
        st.warning("⚠️ Tidak ada data yang sesuai dengan filter. Silakan ubah kriteria filter di sidebar.")
        return

    # PRINSIP 1: Overview First — KPI Cards
    render_kpi_cards(df_filtered)

    # PRINSIP 1: Overview — Grafik utama
    render_overview_charts(df_filtered)

    # PRINSIP 3: Details on Demand — Tabel detail
    render_details_table(df_filtered)

    # Footer
    st.markdown("""
        <div style='text-align: center; margin-top: 2rem; padding: 1rem;
                    border-top: 1px solid rgba(108,99,255,0.2); color: #8892A4; font-size: 0.8rem;'>
            E-Commerce Analytics Dashboard | Streamlit × Plotly | Prinsip HCI Shneiderman
        </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
