
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Judul aplikasi
st.title('Dashboard Analisis Data Bike Sharing')
st.markdown('Proyek Analisis Data oleh **Andes Potipera Sitepu**')

# Mengunggah data
uploaded_file = st.file_uploader("Unggah file CSV dataset penggunaan sepeda", type="csv")

if uploaded_file is not None:
    # Membaca dataset
    data = pd.read_csv(uploaded_file)
    st.write("Dataframe:", data.head())

    # Statistik Deskriptif
    st.subheader("Statistik Deskriptif")
    st.write(data.describe())

    # Visualisasi Penggunaan Sepeda Berdasarkan Waktu
    st.subheader("Penggunaan Sepeda Berdasarkan Waktu")
    jam = st.selectbox("Pilih jam (0-23):", data['hour'].unique())
    filtered_data = data[data['hour'] == jam]

    fig, ax = plt.subplots()
    sns.countplot(data=filtered_data, x='season', ax=ax)
    ax.set_title(f"Penggunaan Sepeda pada Jam {jam} Berdasarkan Musim")
    st.pyplot(fig)

    # Visualisasi Pengaruh Suhu dan Kelembapan
    st.subheader("Pengaruh Suhu dan Kelembapan terhadap Penggunaan Sepeda")
    fig, ax = plt.subplots()
    sns.scatterplot(data=data, x='temp', y='humidity', hue='count', ax=ax)
    ax.set_title("Pengaruh Suhu dan Kelembapan terhadap Jumlah Penggunaan Sepeda")
    st.pyplot(fig)

# Footer
st.markdown("**Kontak:** andessitepu221204@gmail.com")
