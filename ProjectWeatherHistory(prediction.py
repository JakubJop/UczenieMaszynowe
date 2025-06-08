import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np
import warnings

warnings.simplefilter(action='ignore', category=UserWarning)



file_path = r'C:\Users\lucas\OneDrive\Pulpit\top100cities_weather_data.csv'
df = pd.read_csv(file_path)

# czyszczenie danych
df.columns = [col.strip().replace(' ', '_').replace('(', '').replace(')', '').replace('/', '') for col in df.columns]


start_date = pd.to_datetime('2020-01-01')
df['Date'] = [start_date + pd.Timedelta(days=i) for i in range(len(df))]
df['Day_Number'] = (df['Date'] - start_date).dt.days


df_model = df[['Latitude', 'Longitude', 'Wind_Speed_ms', 'Day_Number', 'Temperature_Celsius']].dropna()
X = df_model[['Latitude', 'Longitude', 'Wind_Speed_ms', 'Day_Number']]
y = df_model['Temperature_Celsius']


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


model = LinearRegression()
model.fit(X_train, y_train)


miasto = input("Podaj nazwę miasta: ").strip()
df_city = df[df['City'].str.lower() == miasto.lower()].copy()

if df_city.empty:
    print(f"Brak danych dla miasta: {miasto}")
    exit()

while True:
    date_pred_str = input("Podaj datę prognozy (RRRR-MM-DD) między 2025-06-01 a 2025-12-31: ").strip()
    try:
        date_pred = pd.to_datetime(date_pred_str)
        if pd.Timestamp('2025-06-01') <= date_pred <= pd.Timestamp('2025-12-31'):
            break
        else:
            print("Data musi być między 2025-06-01 a 2025-12-31. Spróbuj ponownie.")
    except:
        print("Niepoprawny format daty. Spróbuj ponownie.")

day_number_pred = (date_pred - start_date).days


lat = df_city.iloc[0]['Latitude']
lon = df_city.iloc[0]['Longitude']
desc = df_city.iloc[0]['Description'] if 'Description' in df_city.columns else 'Brak opisu'
country = df_city.iloc[0]['Country'] if 'Country' in df_city.columns else 'Brak kraju'

wind_speed_pred = df_city['Wind_Speed_ms'].mean() if not df_city['Wind_Speed_ms'].dropna().empty else 0.0


X_future = np.array([[lat, lon, wind_speed_pred, day_number_pred]])
temp_pred_future = model.predict(X_future)[0]


print(f"\nPrognoza dla miasta: {miasto}")
print(f"Kraj: {country}")
print(f"Opis pogody historycznej: {desc}")
print(f"Szerokość geograficzna (Latitude): {lat}")
print(f"Długość geograficzna (Longitude): {lon}")
print(f"Średnia prędkość wiatru: {wind_speed_pred:.2f} m/s")
print(f"Prognozowana temperatura na dzień {date_pred.date()}: {temp_pred_future:.2f} °C")



plt.figure(figsize=(4, 5))
plt.bar(0, temp_pred_future, color='orange', width=0.2)
plt.text(0, temp_pred_future + 0.3, f"{temp_pred_future:.2f} °C", ha='center', fontsize=12)
plt.xlim(-1, 1)
plt.ylim(0, max(temp_pred_future + 5, 10))
plt.xticks([0], [f'{miasto}\n{date_pred.date()}'])
plt.ylabel('Temperatura (°C)')
plt.title(f'Prognozowana temperatura dla miasta {miasto}')
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()
