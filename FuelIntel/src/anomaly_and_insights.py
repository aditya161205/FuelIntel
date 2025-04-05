def detect_anomalies(df):
    # Example rules
    df["Anomaly"] = "None"

    # Fuel theft / drop
    df.loc[df["Fuel Level (L)"] < 10, "Anomaly"] = "Possible fuel theft / drop"

    # High idle time
    df.loc[df["Idle Time(min)"] > 180, "Anomaly"] = "High idle time"

    # Abnormally high fuel consumed per hour
    df.loc[df["Fuel Consumed (L/hr)"] > 50, "Anomaly"] = "Abnormal fuel consumption"

    return df


def generate_insights(df):
    # Add empty tips column
    df["Insight"] = ""

    # Low fuel
    df.loc[df["Fuel Level (L)"] < 15, "Insight"] += "Refuel soon. "

    # Bad fuel efficiency
    df.loc[df["Fuel Efficiency (km/L)"] < 4, "Insight"] += "Improve driving habits. "

    # High engine temperature
    df.loc[df["Engine Temperature(°C)"] > 95, "Insight"] += "Check engine cooling. "

    return df
