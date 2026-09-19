elif page == "⬜ Past Weather":

    st.header("📅 Past Weather")
    st.write("Explore historical weather data.")

    # Load dataset
    past_weather = pd.read_csv("rain_prediction_dataset.csv")

    st.subheader("📊 Historical Weather Records")

    st.dataframe(
        past_weather,
        use_container_width=True
    )

    st.subheader("📈 Weather Trends")

    numeric_columns = past_weather.select_dtypes(
        include="number"
    ).columns.tolist()

    if numeric_columns:

        selected_column = st.selectbox(
            "Select weather parameter",
            numeric_columns
        )

        st.line_chart(
            past_weather[selected_column]
        )