import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import seaborn as sns
import matplotlib.pyplot as plt

# Fungsi untuk membaca data
@st.cache_resource
def read_data():
    hour_df = pd.read_csv('hour.csv')  # Replace with the correct path if needed

    # Mengubah kolom-kolom ke tipe data kategorikal di hour_df
    categorical_columns = ['season', 'yr', 'mnth', 'hr', 'holiday', 'weekday', 'workingday', 'weathersit']
    hour_df[categorical_columns] = hour_df[categorical_columns].astype('category')

    # Fungsi untuk menangani outlier dengan metode imputasi
    def impute_outliers(data, column):
        Q1 = data[column].quantile(0.25)
        Q3 = data[column].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR

        # Imputasi outlier dengan batas atas dan batas bawah
        data[column] = data[column].apply(lambda x: upper_bound if x > upper_bound else (lower_bound if x < lower_bound else x))

    # Imputasi outlier pada kolom 'hum', 'windspeed', 'casual', 'registered', dan 'cnt'
    columns_hour = ['hum', 'windspeed', 'casual', 'registered', 'cnt']
    for column in columns_hour:
        impute_outliers(hour_df, column)

    return hour_df


hour_df = read_data()

# ------------------------
#           TITLE
# ------------------------
st.title("Bike Share Dashboard")

# ------------------------
#          SIDEBAR
# ------------------------
st.sidebar.title("Biodata Author:")
st.sidebar.markdown("**• Nama: Nandana Rifqi Irfansyah**")
st.sidebar.markdown("**• Email: [nandanarifqiirfansyah@gmail.com](mailto:nandanarifqiirfansyah@gmail.com)**")
st.sidebar.markdown("**• Dicoding: [nandanarifqii](https://www.dicoding.com/users/nandanarifqii/)**")

# Menampilkan Dataset Bike Share
if st.sidebar.checkbox("Tampilkan Dataset"):
    st.subheader("Dataset Original")
    st.write(hour_df)

# Menampilkan Deskripsi Ringkasan Dataset
if st.sidebar.checkbox("Tampilkan Ringkasan Dataset"):
    st.subheader("Deskripsi Statistik Dataset")
    st.write(hour_df.describe())

# Informasi tambahan
st.sidebar.markdown('**Season:**')
st.sidebar.markdown('1: Spring, 2: Summer, 3: Fall, 4: Winter')

st.sidebar.markdown('**Weather Situation:**')
st.sidebar.markdown('1: Clear, Few clouds, Partly cloudy')
st.sidebar.markdown('2: Mist + Cloudy, Mist + Broken clouds, Mist')
st.sidebar.markdown('3: Light Snow, Light Rain + Thunderstorm + Scattered clouds')
st.sidebar.markdown('4: Heavy Rain + Ice Pallets, Snow + Fog')

# ------------------------
#       VISUALIZATION
# ------------------------

# Plotting Peminjaman Sepeda Berdasarkan Karateristik Hari & Status
# Menghitung rata-rata peminjaman sepeda casual dan registered pada hari libur dan kerja
mean_rental_casual_weekend = hour_df[hour_df['holiday'] == 1]['casual'].mean()
mean_rental_registered_weekend = hour_df[hour_df['holiday'] == 1]['registered'].mean()
mean_rental_casual_workingday = hour_df[hour_df['workingday'] == 1]['casual'].mean()
mean_rental_registered_workingday = hour_df[hour_df['workingday'] == 1]['registered'].mean()

# Membuat bar chart untuk rata-rata peminjaman sepeda
fig1 = go.Figure(data=[go.Bar(x=['Casual', 'Registered'], y=[mean_rental_casual_workingday, mean_rental_registered_workingday], 
                              marker_color=['blue', 'orange'])])
fig1.update_layout(title='Average Bike Rentals on Weekdays', xaxis_title='Rental Type', yaxis_title='Average Rentals')

fig2 = go.Figure(data=[go.Bar(x=['Casual', 'Registered'], y=[mean_rental_casual_weekend, mean_rental_registered_weekend], 
                              marker_color=['green', 'red'])])
fig2.update_layout(title='Average Bike Rentals on Weekends', xaxis_title='Rental Type', yaxis_title='Average Rentals')

# Menampilkan grafik
st.plotly_chart(fig1, use_container_width=True)
st.plotly_chart(fig2, use_container_width=True)

# Korelasi Situasi Cuaca dan Musim
col1, col2 = st.columns(2)

with col1:
    st.subheader("Korelasi Situasi Cuaca dan Rental Sepeda")
    weather_count_df = hour_df.groupby("weathersit")["cnt"].sum().reset_index()
    fig_weather_count = px.bar(weather_count_df, x="weathersit", y="cnt", title="Situasi Cuaca terhadap Jumlah Rental Sepeda")
    st.plotly_chart(fig_weather_count, use_container_width=True)

with col2:
    st.subheader("Korelasi Musim dan Rental Sepeda")
    season_count_df = hour_df.groupby("season")["cnt"].sum().reset_index()
    fig_season_count = px.bar(season_count_df, x="season", y="cnt", title="Musim terhadap Jumlah Rental Sepeda")
    st.plotly_chart(fig_season_count, use_container_width=True)

# Korelasi Jam, Kecepatan Angin, Suhu, dan Kelembapan
st.subheader("Korelasi Jam, Kecepatan Angin, Suhu dan Kelembapan terhadap Jumlah Rental Sepeda")

plt.figure(figsize=(14, 24))

# Plot untuk hour
plt.subplot(6, 1, 1)
sns.lineplot(data=hour_df, x='hr', y='cnt', color='magenta')
plt.title('Pengaruh Jam Terhadap Jumlah Peminjaman Sepeda')
plt.xlabel('Jam')
plt.ylabel('Jumlah Peminjaman Sepeda')

# Plot untuk windspeed
plt.subplot(6, 1, 2)
sns.lineplot(data=hour_df, x='windspeed', y='cnt', color='red')
plt.title('Pengaruh Kecepatan Angin Terhadap Jumlah Peminjaman Sepeda')

# Plot untuk temp
plt.subplot(6, 1, 3)
sns.lineplot(data=hour_df, x='temp', y='cnt', color='blue')
plt.title('Pengaruh Suhu Terhadap Jumlah Peminjaman Sepeda')

# Plot untuk hum
plt.subplot(6, 1, 4)
sns.lineplot(data=hour_df, x='hum', y='cnt', color='orange')
plt.title('Pengaruh Kelembapan Terhadap Jumlah Peminjaman Sepeda')

# Menampilkan grafik dengan Streamlit
st.pyplot(plt.gcf())

