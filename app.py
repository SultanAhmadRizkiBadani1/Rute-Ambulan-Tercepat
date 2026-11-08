# =====================================================
# STREAMLIT FINAL - AMBULANCE ROUTING BEKASI
# FIXED SESSION STATE | NO TIMEOUT | MAP ALWAYS SHOW
# =====================================================

import streamlit as st
import osmnx as ox
import networkx as nx
import folium
from streamlit_folium import st_folium
from geopy.geocoders import Nominatim
from geopy.exc import GeocoderTimedOut, GeocoderUnavailable
from math import hypot
import time

# =====================================================
# PAGE CONFIG
# =====================================================
st.set_page_config(
    page_title="Ambulance Routing Bekasi",
    page_icon="🚑",
    layout="wide"
)

# =====================================================
# SESSION STATE INIT (INI KUNCI)
# =====================================================
if "run" not in st.session_state:
    st.session_state.run = False

# =====================================================
# CUSTOM CSS (UI PROFESIONAL)
# =====================================================
st.markdown("""
<style>
    .main { background-color: #f9fafb; }
    h1 { color: #b91c1c; }
    .metric-box {
        background-color: white;
        padding: 14px;
        border-radius: 10px;
        box-shadow: 0 2px 6px rgba(0,0,0,0.08);
        margin-bottom: 12px;
        text-align: center;
    }
</style>
""", unsafe_allow_html=True)

# =====================================================
# HEADER
# =====================================================
st.title("🚑 Ambulance Routing System")
st.caption(
    "Perbandingan Algoritma Dijkstra dan A* pada "
    "Mode Normal dan Darurat di Kota & Kabupaten Bekasi"
)

# =====================================================
# SIDEBAR INPUT
# =====================================================
st.sidebar.header("⚙️ Pengaturan")

alamat = st.sidebar.text_input(
    "📍 Lokasi Ambulans",
    "Stasiun Kranji, Bekasi"
)

mode = st.sidebar.selectbox(
    "🚦 Mode Lalu Lintas",
    ["Normal", "Darurat"]
)

algoritma = st.sidebar.selectbox(
    "🧠 Algoritma",
    ["Dijkstra", "A*"]
)

kecepatan = st.sidebar.slider(
    "🚑 Kecepatan Ambulans (km/jam)",
    20, 80, 40, 5
)

# BUTTON (PAKAI SESSION STATE)
if st.sidebar.button("🚀 Proses Rute"):
    st.session_state.run = True

st.sidebar.markdown("---")
st.sidebar.info(
    "Mode **Darurat** mensimulasikan ambulans "
    "yang dapat mengabaikan aturan satu arah."
)

# =====================================================
# CACHE GRAPH
# =====================================================
@st.cache_resource
def load_graph():
    place = "Bekasi, Jawa Barat, Indonesia"

    G_raw = ox.graph_from_place(
        place, network_type="drive", simplify=True
    )

    allowed = [
        "motorway", "trunk", "primary", "secondary",
        "tertiary", "unclassified", "residential"
    ]

    edges_ok = []
    for u, v, k, d in G_raw.edges(keys=True, data=True):
        hw = d.get("highway")
        if isinstance(hw, list):
            hw = hw[0]
        if hw in allowed:
            edges_ok.append((u, v, k))

    G_normal = G_raw.edge_subgraph(edges_ok).copy()
    G_normal = G_normal.subgraph(
        max(nx.strongly_connected_components(G_normal), key=len)
    ).copy()

    G_emergency = G_normal.to_undirected()
    return G_normal, G_emergency

# =====================================================
# CACHE FASKES
# =====================================================
@st.cache_data
def load_faskes():
    place = "Bekasi, Jawa Barat, Indonesia"

    faskes = ox.features_from_place(
        place,
        tags={
            "amenity": ["hospital", "clinic"],
            "healthcare": ["hospital", "clinic"]
        }
    )

    faskes = faskes[
        faskes.geometry.type.isin(["Point", "Polygon", "MultiPolygon"])
    ].copy()

    faskes.loc[
        faskes.geometry.type != "Point",
        "geometry"
    ] = faskes.geometry.centroid

    faskes["name"] = faskes["name"].fillna("Fasilitas Kesehatan")
    return faskes

G_normal, G_emergency = load_graph()
faskes = load_faskes()

# =====================================================
# VALIDASI BUTTON
# =====================================================
if not st.session_state.run:
    st.info("⬅️ Masukkan lokasi dan klik **Proses Rute**")
    st.stop()

# =====================================================
# GEOCODING (PATCHED)
# =====================================================
@st.cache_data(show_spinner=False)
def geocode_address(address):
    geolocator = Nominatim(
        user_agent="ambulance-routing-final",
        timeout=10
    )
    try:
        return geolocator.geocode(address)
    except (GeocoderTimedOut, GeocoderUnavailable):
        return None

loc = geocode_address(alamat + ", Bekasi, Indonesia")

if loc is None:
    st.error(
        "⚠️ Layanan pencarian lokasi sedang sibuk. "
        "Coba beberapa saat lagi."
    )
    st.stop()

ambulance_lat = loc.latitude
ambulance_lon = loc.longitude

# =====================================================
# GRAPH & NODE AWAL
# =====================================================
G_used = G_normal if mode == "Normal" else G_emergency

start_node = ox.distance.nearest_nodes(
    G_normal, ambulance_lon, ambulance_lat
)

start_lat = G_normal.nodes[start_node]["y"]
start_lon = G_normal.nodes[start_node]["x"]

# =====================================================
# PRE-FILTER FASKES (JARAK LURUS)
# =====================================================
def approx_dist(lat1, lon1, lat2, lon2):
    return hypot(lat1 - lat2, lon1 - lon2)

faskes["approx"] = faskes.apply(
    lambda r: approx_dist(
        ambulance_lat, ambulance_lon,
        r.geometry.y, r.geometry.x
    ),
    axis=1
)

faskes_near = faskes.nsmallest(30, "approx")

# =====================================================
# HEURISTIC A*
# =====================================================
def heuristic(u, v):
    x1, y1 = G_used.nodes[u]["x"], G_used.nodes[u]["y"]
    x2, y2 = G_used.nodes[v]["x"], G_used.nodes[v]["y"]
    return ((x1 - x2)**2 + (y1 - y2)**2) ** 0.5

# =====================================================
# PILIH FASKES TERDEKAT
# =====================================================
def select_best_faskes():
    best = {"dist": float("inf")}
    for _, r in faskes_near.iterrows():
        try:
            node = ox.distance.nearest_nodes(
                G_used, r.geometry.x, r.geometry.y
            )
            dist = nx.shortest_path_length(
                G_used, start_node, node, weight="length"
            )
            if dist < best["dist"]:
                best = {
                    "name": r["name"],
                    "lat": r.geometry.y,
                    "lon": r.geometry.x,
                    "dist": dist,
                    "node": node
                }
        except:
            continue
    return best

best = select_best_faskes()

# =====================================================
# ROUTING
# =====================================================
t0 = time.time()

if algoritma == "Dijkstra":
    route = nx.shortest_path(
        G_used, start_node, best["node"], weight="length"
    )
else:
    route = nx.astar_path(
        G_used, start_node, best["node"],
        heuristic=heuristic,
        weight="length"
    )

comp_time = time.time() - t0
eta = (best["dist"] / (kecepatan * 1000 / 3600)) / 60

# =====================================================
# LAYOUT UTAMA
# =====================================================
col1, col2 = st.columns([2, 1])

# ================= MAP =================
with col1:
    st.subheader("🗺️ Peta Rute Ambulans")

    def route_latlon(G, route):
        return [(G.nodes[n]["y"], G.nodes[n]["x"]) for n in route]

    m = folium.Map(
        location=[start_lat, start_lon],
        zoom_start=13
    )

    folium.Marker(
        [start_lat, start_lon],
        popup="🚑 Ambulans",
        icon=folium.Icon(color="blue", icon="ambulance", prefix="fa")
    ).add_to(m)

    folium.Marker(
        [best["lat"], best["lon"]],
        popup=f"🏥 {best['name']}",
        icon=folium.Icon(color="red", icon="plus-sign")
    ).add_to(m)

    folium.PolyLine(
        route_latlon(G_used, route),
        color="red" if mode == "Darurat" else "green",
        weight=5,
        opacity=0.9
    ).add_to(m)

    st_folium(m, height=600, width=900)

# ================= METRICS =================
with col2:
    st.subheader("📊 Hasil")

    st.markdown(
        f"<div class='metric-box'><b>Fasilitas Kesehatan</b><br>{best['name']}</div>",
        unsafe_allow_html=True
    )
    st.markdown(
        f"<div class='metric-box'><b>Jarak Rute</b><br>{best['dist']:.0f} meter</div>",
        unsafe_allow_html=True
    )
    st.markdown(
        f"<div class='metric-box'><b>Estimasi Waktu</b><br>{eta:.2f} menit</div>",
        unsafe_allow_html=True
    )
    st.markdown(
        f"<div class='metric-box'><b>Waktu Komputasi</b><br>{comp_time:.4f} detik</div>",
        unsafe_allow_html=True
    )

# =====================================================
# FOOTER
# =====================================================
st.markdown("---")
st.caption(
    "📌 Sistem simulasi rute ambulans berbasis OpenStreetMap "
    "untuk keperluan penelitian dan akademik."
)
