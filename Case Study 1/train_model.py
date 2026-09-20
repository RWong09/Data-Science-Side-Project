from pathlib import Path

import joblib
import pandas as pd
import xgboost as xgb


BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "flight_fare_prediction_dataset.csv"
MODEL_PATH = BASE_DIR / "xgboost_fare_model.joblib"


def build_features(dataframe: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, float]]:
    data = dataframe.copy()
    departure_datetime = pd.to_datetime(data["Flight Time"].astype(str) + " " + data["Time"])

    data["Is_International"] = (data["InternationalDesc"] == "International").astype(int)
    data["Is_Weekend"] = departure_datetime.dt.dayofweek.isin([4, 5, 6]).astype(int)
    data["Day_Code"] = departure_datetime.dt.dayofweek
    data["Day_of_Month"] = departure_datetime.dt.day
    data["Hour"] = departure_datetime.dt.hour
    data["Minute"] = departure_datetime.dt.minute

    # Preserve the notebook's encoding approach for compatibility with the app.
    flight_num_map = data.groupby("Flight Number")["FarePrice"].mean().to_dict()
    data["Flight_Num_Encoded"] = data["Flight Number"].map(flight_num_map)
    data = pd.get_dummies(data, columns=["Arrival Station"], drop_first=True, dtype=int)

    feature_cols = [
        "Is_International",
        "Is_Weekend",
        "Day_Code",
        "Day_of_Month",
        "Hour",
        "Minute",
        "Flight_Num_Encoded",
    ] + [column for column in data.columns if column.startswith("Arrival Station_")]

    return data[feature_cols].astype(float), flight_num_map


def main() -> None:
    dataframe = pd.read_csv(DATA_PATH)
    features, flight_num_map = build_features(dataframe)
    target = dataframe["FarePrice"].astype(float)
    feature_cols = features.columns.tolist()

    model = xgb.XGBRegressor(
        n_estimators=150,
        learning_rate=0.08,
        max_depth=6,
        random_state=42,
        objective="reg:squarederror",
    )
    model.fit(features, target)

    bundle = {
        "model": model,
        "feature_cols": feature_cols,
        "flight_num_map": flight_num_map,
    }
    joblib.dump(bundle, MODEL_PATH)
    print(f"Saved model bundle to {MODEL_PATH}")
    print(f"Training features: {feature_cols}")


if __name__ == "__main__":
    main()