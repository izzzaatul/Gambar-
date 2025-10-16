# ===========================================
# Image Classification & Object Detection App
# ===========================================
import streamlit as st

# ---------- PAGE CONFIG ----------
st.set_page_config(
    page_title="Image Classification & Object Detection",
    page_icon="🧠",
    layout="wide"
)

# ---------- CUSTOM BACKGROUND (CSS) ----------
page_bg = """
<style>
[data-testid="stAppViewContainer"] {
    background: linear-gradient(to bottom, #d4f8d4 60%, #d4f8d4 80%, #e6d4f8 100%);
    color: #1b1b1b;
    font-family: "Poppins", sans-serif;
}

[data-testid="stHeader"] {
    background: rgba(0,0,0,0);
}

h1, h2, h3 {
    font-weight: 600;
    color: #1b4332;
}

p {
    font-size: 16px;
    line-height: 1.6;
}

img {
    border-radius: 20px;
    box-shadow: 0 4px 10px rgba(0,0,0,0.15);
}
</style>
"""
st.markdown(page_bg, unsafe_allow_html=True)

# ---------- HEADER ----------
st.markdown("<h1 style='text-align: center;'>Image Classification & Object Detection</h1>", unsafe_allow_html=True)
st.write("")

# =====================================================
# BIG CATS SECTION
# =====================================================
st.markdown("## 🦁 Big Cats")

col1, col2 = st.columns([1, 1.8])

with col1:
    st.image([
        "https://cdn-icons-png.flaticon.com/512/616/616408.png",  # Lion cartoon
        "https://cdn-icons-png.flaticon.com/512/616/616425.png",  # Leopard cartoon
        "https://cdn-icons-png.flaticon.com/512/616/616430.png"   # Tiger cartoon
    ], caption=["Lion", "Leopard", "Tiger"], width=180)

with col2:
    st.markdown("""
    **Big cats** digunakan untuk menyebut kelompok kucing besar yang termasuk dalam keluarga *Felidae* dan umumnya merupakan predator puncak di alam liar.  
    Hewan-hewan ini memiliki tubuh besar, kekuatan luar biasa, serta kemampuan berburu yang sangat efisien.  
    Contoh yang termasuk kategori *big cats* antara lain **singa, harimau, macan tutul, jaguar, cheetah, puma**, dan **snow leopard**.  

    Sebagian besar dari mereka berasal dari genus *Panthera*, yang dikenal karena kemampuannya untuk mengaum (*roar*) berkat struktur pita suara khusus pada laringnya.  
    *Big cats* hidup di berbagai habitat seperti hutan hujan, sabana, pegunungan, hingga padang rumput, dan berperan penting dalam menjaga keseimbangan ekosistem karena memangsa herbivora dan mencegah populasi hewan mangsa menjadi berlebihan.  
    Mereka merupakan simbol kekuatan, keanggunan, dan keindahan alam liar yang sering kali menjadi ikon konservasi satwa dunia.
    """)

# =====================================================
# DOMESTIC CATS SECTION
# =====================================================
st.markdown("---")
st.markdown("## 🐱 Cats")

col3, col4 = st.columns([1, 1.8])

with col3:
    st.image([
        "https://cdn-icons-png.flaticon.com/512/616/616408.png",  # Cute cat
        "https://cdn-icons-png.flaticon.com/512/616/616408.png"   # Another cat
    ], caption=["Domestic Cat 1", "Domestic Cat 2"], width=180)

with col4:
    st.markdown("""
    **Cats** merujuk pada semua anggota keluarga *Felidae*, namun dalam penggunaan sehari-hari lebih sering digunakan untuk menyebut **kucing domestik (*Felis catus*)**.  
    Kucing domestik adalah keturunan dari kucing liar kecil yang telah mengalami proses domestikasi oleh manusia selama ribuan tahun.  

    Berbeda dengan *big cats*, kucing domestik berukuran kecil, bersifat jinak, dan hidup berdampingan dengan manusia sebagai hewan peliharaan.
    """)

# =====================================================
# FOOTER
# =====================================================
st.markdown("---")
st.markdown("<p style='text-align:center; font-size:14px; color:#444;'>© 2025 Image Classification Dashboard — created with Streamlit</p>", unsafe_allow_html=True)
