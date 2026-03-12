from flask import Flask, request, jsonify, render_template
import joblib, requests, os
from datetime import datetime, timedelta, timezone

app = Flask(__name__)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "decision_tree_banjir.pkl")

# Load model
model_data = joblib.load("decision_tree_banjir.pkl")
model = model_data["model"]
feature_names = model_data["feature_names"]

# Mapping label
label_mapping = {0: "aman", 1: "siaga", 2: "banjir"}


# Fungsi akumulasi curah hujan harian
def get_daily_rain_mm(forecast_data):
    jakarta_tz = timezone(timedelta(hours=7))
    today = datetime.now(jakarta_tz).date()

    rain_today = 0.0
    for item in forecast_data.get("list", []):
        dt = datetime.fromtimestamp(item["dt"], jakarta_tz).date()
        if dt == today:
            rain_today += item.get("rain", {}).get("3h", 0.0)
    return rain_today


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["GET"])
def predict():
    lat = request.args.get("lat")
    lon = request.args.get("lon")
    
    manual_rain = request.args.get("rain")
    manual_humidity = request.args.get("humidity")

    if not lat or not lon:
        return jsonify({"error": "Latitude & Longitude harus diisi"}), 400

    OPENWEATHER_API_KEY = os.getenv("OPENWEATHER_API_KEY")
    url = f"https://api.openweathermap.org/data/2.5/forecast?lat={lat}&lon={lon}&appid={OPENWEATHER_API_KEY}&units=metric"
    response = requests.get(url).json()

    if "list" not in response or len(response["list"]) == 0:
        return jsonify({"error": "Data cuaca tidak ditemukan"}), 404

    # Pakai data 3 jam pertama (list[0]) buat parameter selain rain
    first_data = response["list"][0]
    main_data = first_data.get("main", {})
    wind_data = first_data.get("wind", {})

    humidity = float(manual_humidity) if manual_humidity is not None else main_data.get("humidity", 0)
    temp_max = main_data.get("temp_max", 0)
    temp_min = main_data.get("temp_min", 0)
    temp_mean = main_data.get("temp", 0)
    dew_point = temp_min - 2 if temp_min != 0 else 0
    wind_max = wind_data.get("speed", 0)

    # Curah hujan
    rain = float(manual_rain) if manual_rain is not None else get_daily_rain_mm(response)

    # Prediksi
    features = [[humidity, temp_max, temp_min, temp_mean, dew_point, wind_max, rain]]
    try:
        prediction = model.predict(features)[0]
        prediction_label = label_mapping.get(prediction, "unknown")
    except Exception as e:
        return jsonify({"error": f"Gagal prediksi: {str(e)}"}), 500

    return jsonify({
        "city": response.get("city", {}).get("name", "unknown"),
        "lat": lat,
        "lon": lon,
        "parameters": {
            "Kelembapan": humidity,
            "Temperatur Maksimum": temp_max,
            "Temperatur Minimum": temp_min,
            "Temperatur Rata-rata": temp_mean,
            "Titik Embun": dew_point,
            "Kecepatan Angin Maksimum": wind_max,
            "Curah Hujan": rain
        },
        "prediction": prediction_label
    })


if __name__ == "__main__":
    app.run(debug=True)