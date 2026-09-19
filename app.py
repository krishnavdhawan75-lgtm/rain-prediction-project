import streamlit as st
import pandas as pd
import numpy as np
import requests
import plotly.express as px
from datetime import datetime, date, timedelta
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
# =========================================================
# PAGE CONFIGURATION
# =========================================================

st.set_page_config(
    page_title="Rain Prediction System",
    page_icon="🌧️",
    layout="wide"
)

# =========================================================
# CUSTOM CSS
# =========================================================

st.markdown(
    """
    <style>
    .main {
        background-color: #f5f9ff;
    }

    h1, h2, h3 {
        color: #12355b;
    }

    .stMetric {
    background-color: white;
    padding: 15px;
    border-radius: 10px;
    color: #12355b;
}

.stMetric label,
.stMetric [data-testid="stMetricValue"],
.stMetric [data-testid="stMetricDelta"] {
    color: #12355b !important;
}
    </style>
    """,
    unsafe_allow_html=True
)

# =========================================================
# 10 CITY DATA
# =========================================================

CITIES = {
    "Delhi": {
        "latitude": 28.6139,
        "longitude": 77.2090
    },
    "Mumbai": {
        "latitude": 19.0760,
        "longitude": 72.8777
    },
    "Bengaluru": {
        "latitude": 12.9716,
        "longitude": 77.5946
    },
    "Chennai": {
        "latitude": 13.0827,
        "longitude": 80.2707
    },
    "Kolkata": {
        "latitude": 22.5726,
        "longitude": 88.3639
    },
    "Hyderabad": {
        "latitude": 17.3850,
        "longitude": 78.4867
    },
    "Pune": {
        "latitude": 18.5204,
        "longitude": 73.8567
    },
    "Ahmedabad": {
        "latitude": 23.0225,
        "longitude": 72.5714
    },
    "Jaipur": {
        "latitude": 26.9124,
        "longitude": 75.7873
    },
    "Lucknow": {
        "latitude": 26.8467,
        "longitude": 80.9462
    }
}

# =========================================================
# SIDEBAR AND CITY SELECTION
# =========================================================

st.sidebar.title("🌧️ Rain Prediction")

selected_city = st.sidebar.selectbox(
    "Select Your City",
    options=list(CITIES.keys()),
    index=0
)

latitude = CITIES[selected_city]["latitude"]
longitude = CITIES[selected_city]["longitude"]

st.sidebar.success(
    f"Selected City: {selected_city}"
)

st.sidebar.write(f"Latitude: {latitude}")
st.sidebar.write(f"Longitude: {longitude}")

page = st.sidebar.radio(
    "Navigation",
    [
        "🏠 Home",
        "🌧️ Rain Prediction",
        "📊 Data Analysis",
        "🤖 AI Model",
        "🌍 Sustainability",
        "📅 Past Weather",
        "🌤️ Current Weather",
        "🔮 Future Weather",
        "ℹ️ About Project"
    ]
)

# =========================================================
# LOAD DATASET
# =========================================================

@st.cache_data
def load_dataset():

    file_names = [
        "rain_prediction_dataset.csv",
        "Rainfall.csv",
        "weather.csv",
        "dataset.csv"
    ]

    for file_name in file_names:

        try:
            loaded_data = pd.read_csv(file_name)
            return loaded_data, file_name

        except FileNotFoundError:
            continue

    return None, None


data, dataset_name = load_dataset()

# =========================================================
# CURRENT WEATHER API
# =========================================================

@st.cache_data(ttl=1800)
def get_current_weather(latitude, longitude):

    url = "https://api.open-meteo.com/v1/forecast"

    parameters = {
        "latitude": latitude,
        "longitude": longitude,
        "current": [
            "temperature_2m",
            "relative_humidity_2m",
            "precipitation",
            "rain",
            "wind_speed_10m",
            "weather_code"
        ],
        "timezone": "auto"
    }

    response = requests.get(
        url,
        params=parameters,
        timeout=30
    )

    response.raise_for_status()

    return response.json()

# =========================================================
# FUTURE WEATHER API
# =========================================================

@st.cache_data(ttl=1800)
def get_future_weather(latitude, longitude):

    url = "https://api.open-meteo.com/v1/forecast"

    parameters = {
        "latitude": latitude,
        "longitude": longitude,
        "daily": [
            "temperature_2m_max",
            "temperature_2m_min",
            "precipitation_sum",
            "rain_sum",
            "precipitation_probability_max",
            "weather_code"
        ],
        "forecast_days": 7,
        "timezone": "auto"
    }

    response = requests.get(
        url,
        params=parameters,
        timeout=30
    )

    response.raise_for_status()

    return response.json()

# =========================================================
# PAST WEATHER API
# =========================================================

@st.cache_data(ttl=3600)
def get_past_weather(latitude, longitude):

    today = date.today()

    start_date = today - timedelta(days=365)
    end_date = today - timedelta(days=1)

    url = "https://archive-api.open-meteo.com/v1/archive"

    parameters = {
        "latitude": latitude,
        "longitude": longitude,
        "start_date": start_date.strftime("%Y-%m-%d"),
        "end_date": end_date.strftime("%Y-%m-%d"),
        "daily": [
            "temperature_2m_max",
            "temperature_2m_min",
            "precipitation_sum",
            "rain_sum",
            "wind_speed_10m_max"
        ],
        "timezone": "auto"
    }

    response = requests.get(
        url,
        params=parameters,
        timeout=60
    )

    response.raise_for_status()

    return response.json()

# =========================================================
# WEATHER DESCRIPTION
# =========================================================

def weather_description(weather_code):

    weather_codes = {
        0: "Clear sky",
        1: "Mainly clear",
        2: "Partly cloudy",
        3: "Overcast",
        45: "Fog",
        48: "Rime fog",
        51: "Light drizzle",
        53: "Moderate drizzle",
        55: "Dense drizzle",
        61: "Slight rain",
        63: "Moderate rain",
        65: "Heavy rain",
        71: "Slight snow",
        73: "Moderate snow",
        75: "Heavy snow",
        80: "Slight rain showers",
        81: "Moderate rain showers",
        82: "Heavy rain showers",
        95: "Thunderstorm"
    }

    return weather_codes.get(
        int(weather_code),
        "Unknown weather"
    )

# =========================================================
# HOME PAGE
# =========================================================

if page == "🏠 Home":

    st.title(
        "🌧️ Rain Prediction and Weather Monitoring System"
    )

    st.write(
        "Welcome to the Rain Prediction System."
    )

    st.write(
        "Select any one of the 10 cities from the sidebar "
        "to view weather information."
    )

    st.divider()

    st.subheader(
        f"📍 Selected City: {selected_city}"
    )

    city_information = pd.DataFrame(
        {
            "City": [selected_city],
            "Latitude": [latitude],
            "Longitude": [longitude]
        }
    )

    st.dataframe(
        city_information,
        use_container_width=True,
        hide_index=True
    )

    col1, col2, col3 = st.columns(3)

    with col1:
        st.info("📅 Past Weather")
        st.write(
            "View weather data from the previous year."
        )

    with col2:
        st.success("🌤️ Current Weather")
        st.write(
            "View current weather conditions."
        )

    with col3:
        st.warning("🔮 Future Weather")
        st.write(
            "View the next 7 days of forecast data."
        )

    st.subheader("🗺️ Selected City Location")

    map_data = pd.DataFrame(
        {
            "latitude": [latitude],
            "longitude": [longitude]
        }
    )

    st.map(map_data)

# =========================================================
# AI RAIN PREDICTION PAGE
# =========================================================

elif page == "🌧️ Rain Prediction":

    st.title(
        f"🤖 AI Rain Prediction - {selected_city}"
    )

    st.write(
        "Enter weather values below to predict rain "
        "using a Random Forest Machine Learning model."
    )

    if data is None:

        st.error(
            "Dataset not found. Please keep your CSV file "
            "in the same folder as app.py."
        )

    else:

        st.success(
            f"Dataset loaded: {dataset_name}"
        )

        # -------------------------------------------------
        # SELECT TARGET COLUMN
        # -------------------------------------------------

        st.subheader(
            "🎯 Select the Rain Target Column"
        )

        possible_target_columns = [
            column
            for column in data.columns
            if any(
                word in column.lower()
                for word in [
                    "rain",
                    "rainfall",
                    "precipitation"
                ]
            )
        ]

        if len(possible_target_columns) == 0:

            st.warning(
                "No rain-related column was automatically found."
            )

            target_column = st.selectbox(
                "Select the target column manually",
                data.columns
            )

        else:

            target_column = st.selectbox(
                "Select the target column",
                possible_target_columns
            )

        st.write(
            f"Selected target: **{target_column}**"
        )

        # -------------------------------------------------
        # PREPARE TARGET COLUMN
        # -------------------------------------------------

        target_values = data[target_column]

        if pd.api.types.is_numeric_dtype(
            target_values
        ):

            numeric_target = pd.to_numeric(
                target_values,
                errors="coerce"
            )

            unique_values = (
                numeric_target.dropna().unique()
            )

            if len(unique_values) > 2:

                # Rainfall greater than zero = rain
                y = (
                    numeric_target
                    .fillna(0)
                    .gt(0)
                    .astype(int)
                )

            else:

                y = numeric_target

        else:

            target_text = (
                target_values
                .astype(str)
                .str.strip()
                .str.lower()
            )

            rain_words = [
                "yes",
                "rain",
                "rainy",
                "true",
                "1",
                "y"
            ]

            y = target_text.apply(
                lambda value:
                1
                if value in rain_words
                else 0
            )

        # -------------------------------------------------
        # SELECT NUMERIC FEATURES
        # -------------------------------------------------

        feature_data = data.drop(
            columns=[target_column]
        )

        numeric_features = (
            feature_data
            .select_dtypes(include=np.number)
            .columns
            .tolist()
        )

        if len(numeric_features) == 0:

            st.error(
                "No numeric weather features were found "
                "in your dataset."
            )

            st.info(
                "Your dataset needs numeric columns such as "
                "temperature, humidity, pressure, or wind speed."
            )

        else:

            X = feature_data[numeric_features].copy()

            X = X.replace(
                [np.inf, -np.inf],
                np.nan
            )

            y = pd.to_numeric(
                y,
                errors="coerce"
            )

            valid_rows = y.notna()

            X = X.loc[valid_rows]
            y = y.loc[valid_rows].astype(int)

            # Remove columns that contain no usable values
            usable_features = [
                column
                for column in X.columns
                if X[column].notna().any()
            ]

            X = X[usable_features]

            if len(usable_features) == 0:

                st.error(
                    "No usable numeric features are available."
                )

            elif y.nunique() < 2:

                st.error(
                    "The target column must contain both "
                    "rain and no-rain values."
                )

            elif len(X) < 10:

                st.error(
                    "Your dataset has too few valid records "
                    "to train the model reliably."
                )

            else:

                st.subheader(
                    "📊 Training Information"
                )

                st.write(
                    f"Number of records: {len(X)}"
                )

                st.write(
                    f"Features used: {usable_features}"
                )

                # -------------------------------------------------
                # IMPUTE MISSING VALUES
                # -------------------------------------------------

                imputer = SimpleImputer(
                    strategy="median"
                )

                X_imputed = imputer.fit_transform(X)

                # -------------------------------------------------
                # TRAIN TEST SPLIT
                # -------------------------------------------------

                try:

                    X_train, X_test, y_train, y_test = (
                        train_test_split(
                            X_imputed,
                            y,
                            test_size=0.2,
                            random_state=42,
                            stratify=y
                        )
                    )

                except ValueError:

                    X_train, X_test, y_train, y_test = (
                        train_test_split(
                            X_imputed,
                            y,
                            test_size=0.2,
                            random_state=42
                        )
                    )

                # -------------------------------------------------
                # TRAIN RANDOM FOREST MODEL
                # -------------------------------------------------

                model = RandomForestClassifier(
                    n_estimators=100,
                    random_state=42,
                    class_weight="balanced"
                )

                model.fit(
                    X_train,
                    y_train
                )

                # -------------------------------------------------
                # MODEL ACCURACY
                # -------------------------------------------------

                test_predictions = model.predict(
                    X_test
                )

                accuracy = accuracy_score(
                    y_test,
                    test_predictions
                )

                st.subheader(
                    "📈 Model Performance"
                )

                st.metric(
                    "Model Accuracy",
                    f"{accuracy * 100:.2f}%"
                )

                # -------------------------------------------------
                # USER INPUT SECTION
                # -------------------------------------------------

                st.divider()

                st.subheader(
                    "📝 Enter Weather Values"
                )

                st.write(
                    "Enter the values for the weather features "
                    "used by the AI model."
                )

                user_values = []

                input_columns = st.columns(2)

                for index, feature in enumerate(
                    usable_features
                ):

                    feature_values = pd.to_numeric(
                        X[feature],
                        errors="coerce"
                    ).dropna()

                    if len(feature_values) == 0:

                        default_value = 0.0
                        minimum_value = -1000.0
                        maximum_value = 1000.0

                    else:

                        default_value = float(
                            feature_values.median()
                        )

                        minimum_value = float(
                            feature_values.min()
                        )

                        maximum_value = float(
                            feature_values.max()
                        )

                        if minimum_value == maximum_value:

                            minimum_value -= 1
                            maximum_value += 1

                    with input_columns[index % 2]:

                        user_value = st.number_input(
                            label=f"{feature}",
                            min_value=minimum_value,
                            max_value=maximum_value,
                            value=default_value,
                            key=f"rain_input_{index}"
                        )

                    user_values.append(
                        user_value
                    )

                # -------------------------------------------------
                # PREDICT RAIN BUTTON
                # -------------------------------------------------

                st.divider()

                if st.button(
                    "🔮 Predict Rain",
                    type="primary",
                    use_container_width=True
                ):

                    user_input = np.array(
                        [user_values]
                    )

                    user_input_imputed = (
                        imputer.transform(user_input)
                    )

                    prediction = model.predict(
                        user_input_imputed
                    )[0]

                    probabilities = model.predict_proba(
                        user_input_imputed
                    )[0]

                    class_labels = list(
                        model.classes_
                    )

                    rain_probability = 0.0
                    no_rain_probability = 0.0

                    for class_label, probability in zip(
                        class_labels,
                        probabilities
                    ):

                        if int(class_label) == 1:

                            rain_probability = (
                                probability * 100
                            )

                        else:

                            no_rain_probability = (
                                probability * 100
                            )

                    # -------------------------------------------------
                    # DISPLAY RESULT
                    # -------------------------------------------------

                    st.subheader(
                        "🌧️ AI Prediction Result"
                    )

                    result_col1, result_col2 = st.columns(2)

                    with result_col1:

                        st.metric(
                            "Rain Probability",
                            f"{rain_probability:.2f}%"
                        )

                    with result_col2:

                        st.metric(
                            "No Rain Probability",
                            f"{no_rain_probability:.2f}%"
                        )

                    st.progress(
                        int(round(rain_probability))
                    )

                    if int(prediction) == 1:

                        st.error(
                            "🌧️ Prediction: Rain is expected."
                        )

                    else:
                        st.metric("🌧️ Rain Probability", f"{rain_probability:.2f}%")
                        st.metric("☀️ No Rain Probability", f"{no_rain_probability:.2f}%")
                        st.success(
                            "☀️ Prediction: No rain is expected."
                        )

                    st.info(
                        "The probability is estimated by the "
                        "trained Random Forest model. It is not "
                        "a guaranteed weather forecast."
                    )

elif page == "📊 Data Analysis":

    st.title("📊 Data Analysis")

    if data is None:

        st.warning(
            "Dataset not found. Please keep your CSV file "
            "in the same folder as app.py."
        )

    else:

        st.success(
            f"Dataset loaded: {dataset_name}"
        )

        st.subheader("Dataset Preview")

        st.dataframe(
            data.head(20),
            use_container_width=True
        )

        st.subheader("Dataset Information")

        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric(
                "Total Rows",
                data.shape[0]
            )

        with col2:
            st.metric(
                "Total Columns",
                data.shape[1]
            )

        with col3:
            st.metric(
                "Missing Values",
                int(data.isnull().sum().sum())
            )

        st.subheader("Column Names")

        st.write(
            list(data.columns)
        )

        st.subheader("Statistical Summary")

        st.dataframe(
            data.describe(include="all").transpose(),
            use_container_width=True
        )

        numeric_columns = data.select_dtypes(
            include=np.number
        ).columns.tolist()

        if len(numeric_columns) > 0:

            st.subheader(
                "Numeric Column Visualization"
            )

            selected_column = st.selectbox(
                "Select a numeric column",
                numeric_columns
            )

            chart = px.histogram(
                data,
                x=selected_column,
                title=f"Distribution of {selected_column}"
            )

            st.plotly_chart(
                chart,
                use_container_width=True
            )

# =========================================================
# AI MODEL PAGE
# =========================================================

elif page == "🤖 AI Model":

    st.title("🤖 AI Model")

    st.write(
        "This section describes the machine learning "
        "component of the project."
    )

    if data is None:

        st.warning(
            "Dataset not found. Add your dataset CSV file "
            "to the project folder."
        )

    else:

        st.success(
            "Dataset is available."
        )

        st.write(
            f"Dataset contains {data.shape[0]} rows "
            f"and {data.shape[1]} columns."
        )

        st.write(
            "Possible machine learning features include "
            "temperature, humidity, pressure, wind speed, "
            "and rainfall."
        )

        st.info(
            "Connect your trained machine learning model "
            "here if you have a saved model file."
        )

# =========================================================
# SUSTAINABILITY PAGE
# =========================================================

elif page == "🌍 Sustainability":

    st.title("🌍 Sustainability")

    st.write(
        "Rain prediction and weather monitoring can "
        "support sustainable development."
    )

    st.subheader("Benefits")

    st.write(
        "🌱 Helps farmers plan irrigation."
    )

    st.write(
        "💧 Supports water resource management."
    )

    st.write(
        "🌾 Helps reduce crop damage."
    )

    st.write(
        "🏙️ Supports preparation for heavy rainfall."
    )

    st.write(
        "🌍 Helps communities prepare for weather changes."
    )

# =========================================================
# PAST WEATHER PAGE
# =========================================================

elif page == "📅 Past Weather":

    st.title(
        f"📅 Past Weather - {selected_city}"
    )

    st.write(
        "Historical weather information for approximately "
        "the previous 365 days."
    )

    with st.spinner(
        "Loading past weather data..."
    ):

        try:

            past_weather = get_past_weather(
                latitude,
                longitude
            )

            daily_data = past_weather.get(
                "daily",
                {}
            )

            if not daily_data:

                st.warning(
                    "No past weather data was found."
                )

            else:

                past_dataframe = pd.DataFrame(
                    daily_data
                )

                past_dataframe["time"] = pd.to_datetime(
                    past_dataframe["time"]
                )

                st.success(
                    f"Loaded {len(past_dataframe)} days "
                    "of historical weather data."
                )

                st.subheader(
                    "Historical Weather Dataset"
                )

                st.dataframe(
                    past_dataframe,
                    use_container_width=True
                )

                st.subheader(
                    "Historical Temperature"
                )

                temperature_columns = []

                if "temperature_2m_max" in past_dataframe.columns:

                    temperature_columns.append(
                        "temperature_2m_max"
                    )

                if "temperature_2m_min" in past_dataframe.columns:

                    temperature_columns.append(
                        "temperature_2m_min"
                    )

                if len(temperature_columns) > 0:

                    temperature_chart = px.line(
                        past_dataframe,
                        x="time",
                        y=temperature_columns,
                        title="Temperature During Previous Year"
                    )

                    st.plotly_chart(
                        temperature_chart,
                        use_container_width=True
                    )

                st.subheader(
                    "Historical Rainfall"
                )

                rainfall_column = None

                if "rain_sum" in past_dataframe.columns:

                    rainfall_column = "rain_sum"

                elif "precipitation_sum" in past_dataframe.columns:

                    rainfall_column = "precipitation_sum"

                if rainfall_column is not None:

                    rainfall_chart = px.line(
                        past_dataframe,
                        x="time",
                        y=rainfall_column,
                        title="Rainfall During Previous Year"
                    )

                    st.plotly_chart(
                        rainfall_chart,
                        use_container_width=True
                    )

                csv_file = past_dataframe.to_csv(
                    index=False
                ).encode("utf-8")

                st.download_button(
                    label="Download Past Weather CSV",
                    data=csv_file,
                    file_name=(
                        f"{selected_city}_past_weather.csv"
                    ),
                    mime="text/csv"
                )

        except Exception as error:

            st.error(
                f"Unable to load past weather data: {error}"
            )

# =========================================================
# CURRENT WEATHER PAGE
# =========================================================

elif page == "🌤️ Current Weather":

    st.title(
        f"🌤️ Current Weather - {selected_city}"
    )

    with st.spinner(
        "Loading current weather..."
    ):

        try:

            weather_data = get_current_weather(
                latitude,
                longitude
            )

            current = weather_data["current"]

            temperature = current.get(
                "temperature_2m",
                "N/A"
            )

            humidity = current.get(
                "relative_humidity_2m",
                "N/A"
            )

            precipitation = current.get(
                "precipitation",
                "N/A"
            )

            rain = current.get(
                "rain",
                "N/A"
            )

            wind_speed = current.get(
                "wind_speed_10m",
                "N/A"
            )

            weather_code = current.get(
                "weather_code",
                -1
            )

            weather_text = weather_description(
                weather_code
            )

            col1, col2, col3 = st.columns(3)

            with col1:

                st.metric(
                    "Temperature",
                    f"{temperature} °C"
                )

            with col2:

                st.metric(
                    "Humidity",
                    f"{humidity} %"
                )

            with col3:

                st.metric(
                    "Wind Speed",
                    f"{wind_speed} km/h"
                )

            col4, col5, col6 = st.columns(3)

            with col4:

                st.metric(
                    "Precipitation",
                    f"{precipitation} mm"
                )

            with col5:

                st.metric(
                    "Rain",
                    f"{rain} mm"
                )

            with col6:

                st.metric(
                    "Weather",
                    weather_text
                )

            st.subheader(
                "Current Weather Details"
            )

            st.json(current)

        except Exception as error:

            st.error(
                f"Unable to load current weather: {error}"
            )

# =========================================================
# FUTURE WEATHER PAGE
# =========================================================

elif page == "🔮 Future Weather":

    st.title(
        f"🔮 Future Weather - {selected_city}"
    )

    st.write(
        "Weather forecast for the next 7 days."
    )

    with st.spinner(
        "Loading future weather forecast..."
    ):

        try:

            future_weather = get_future_weather(
                latitude,
                longitude
            )

            daily_data = future_weather.get(
                "daily",
                {}
            )

            if not daily_data:

                st.warning(
                    "No future weather data was found."
                )

            else:

                future_dataframe = pd.DataFrame(
                    daily_data
                )

                future_dataframe["time"] = pd.to_datetime(
                    future_dataframe["time"]
                )

                st.subheader(
                    "7-Day Weather Forecast"
                )

                display_columns = [
                    "time",
                    "temperature_2m_max",
                    "temperature_2m_min",
                    "precipitation_sum",
                    "rain_sum",
                    "precipitation_probability_max",
                    "weather_code"
                ]

                available_columns = [
                    column
                    for column in display_columns
                    if column in future_dataframe.columns
                ]

                st.dataframe(
                    future_dataframe[available_columns],
                    use_container_width=True
                )

                st.subheader(
                    "🌡️ Future Temperature Forecast"
                )

                temperature_columns = []

                if "temperature_2m_max" in future_dataframe.columns:

                    temperature_columns.append(
                        "temperature_2m_max"
                    )

                if "temperature_2m_min" in future_dataframe.columns:

                    temperature_columns.append(
                        "temperature_2m_min"
                    )

                if len(temperature_columns) > 0:

                    temperature_chart = px.line(
                        future_dataframe,
                        x="time",
                        y=temperature_columns,
                        markers=True,
                        title="7-Day Temperature Forecast"
                    )

                    st.plotly_chart(
                        temperature_chart,
                        use_container_width=True
                    )

                st.subheader(
                    "🌧️ Future Rainfall Forecast"
                )

                rainfall_column = None

                if "rain_sum" in future_dataframe.columns:

                    rainfall_column = "rain_sum"

                elif "precipitation_sum" in future_dataframe.columns:

                    rainfall_column = "precipitation_sum"

                if rainfall_column is not None:

                    rainfall_chart = px.bar(
                        future_dataframe,
                        x="time",
                        y=rainfall_column,
                        title="7-Day Rainfall Forecast"
                    )

                    st.plotly_chart(
                        rainfall_chart,
                        use_container_width=True
                    )

                st.subheader(
                    "☔ Rain Probability Forecast"
                )

                if (
                    "precipitation_probability_max"
                    in future_dataframe.columns
                ):

                    probability_chart = px.line(
                        future_dataframe,
                        x="time",
                        y="precipitation_probability_max",
                        markers=True,
                        title="Probability of Rain"
                    )

                    st.plotly_chart(
                        probability_chart,
                        use_container_width=True
                    )

                st.subheader(
                    "📅 Daily Forecast Summary"
                )

                for _, row in future_dataframe.iterrows():

                    forecast_date = row["time"].strftime(
                        "%d-%m-%Y"
                    )

                    maximum_temperature = row.get(
                        "temperature_2m_max",
                        "N/A"
                    )

                    minimum_temperature = row.get(
                        "temperature_2m_min",
                        "N/A"
                    )

                    rainfall = row.get(
                        "rain_sum",
                        row.get(
                            "precipitation_sum",
                            "N/A"
                        )
                    )

                    probability = row.get(
                        "precipitation_probability_max",
                        "N/A"
                    )

                    with st.expander(
                        f"📅 {forecast_date}"
                    ):

                        st.write(
                            f"Maximum Temperature: "
                            f"{maximum_temperature} °C"
                        )

                        st.write(
                            f"Minimum Temperature: "
                            f"{minimum_temperature} °C"
                        )

                        st.write(
                            f"Rainfall: {rainfall} mm"
                        )

                        st.write(
                            f"Rain Probability: "
                            f"{probability} %"
                        )

                csv_file = future_dataframe.to_csv(
                    index=False
                ).encode("utf-8")

                st.download_button(
                    label="Download Future Weather CSV",
                    data=csv_file,
                    file_name=(
                        f"{selected_city}_future_weather.csv"
                    ),
                    mime="text/csv"
                )

        except Exception as error:

            st.error(
                f"Unable to load future weather data: {error}"
            )

# =========================================================
# ABOUT PROJECT PAGE
# =========================================================

elif page == "ℹ️ About Project":

    st.title("ℹ️ About Project")

    st.subheader(
        "Project Name"
    )

    st.write(
        "Rain Prediction and Weather Monitoring System"
    )

    st.subheader(
        "Project Objective"
    )

    st.write(
        "The objective of this project is to provide "
        "historical, current, and future weather information "
        "for selected cities."
    )

    st.subheader(
        "Technologies Used"
    )

    st.write("🐍 Python")
    st.write("🎨 Streamlit")
    st.write("📊 Pandas")
    st.write("📈 Plotly")
    st.write("🌐 Open-Meteo Weather API")
    st.write("🤖 Machine Learning")

    st.subheader(
        "Main Features"
    )

    st.write("• Selection of 10 Indian cities")
    st.write("• Past weather data")
    st.write("• Current weather information")
    st.write("• Seven-day future weather forecast")
    st.write("• Weather data visualization")
    st.write("• Basic rain prediction indication")

# =========================================================
# FOOTER
# =========================================================

st.sidebar.divider()

st.sidebar.caption(
    "🌧️ Rain Prediction System"
)