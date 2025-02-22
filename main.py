from classes import WinProbabilityEstimator, PointsPredictor, MarginPredictor
from functions import (
    prepare_prediction_data,
    fetch_and_merge_basketball_data,
    process_data_main,
    filter_relevant_columns_main,
    create_predictions_df,
    replace_nan_with_value
)
import joblib

def main():
    # Input parameters (to be set by the user)
    league = input("Enter the league ID: ")  # Manually input the league ID
    api_key = input("Enter your API key: ")
    api_host = "v1.basketball.api-sports.io"

    # Step 1: Fetch matches data
    print("Fetching test data for prediction...")
    testdata_for_prediction = prepare_prediction_data(league, api_host, api_key, fetch_and_merge_basketball_data)

    # Step 2: Prepare data for prediction
    print("Processing data for prediction...")
    data_test = testdata_for_prediction.copy()
    data_test_no_teams, list_home, list_away = process_data_main(data_test)
    data_test_clean = filter_relevant_columns_main(data_test_no_teams)
    data_test_clean=replace_nan_with_value(data_test_clean,0.5)
    print("Loading models and scalers...")
    points_predictor_model = joblib.load('models/points_predictor_model.pkl')
    win_model = joblib.load('models/win_model (1).pkl')
    win_scaler = joblib.load('models/win_scaler.pkl')
    margin_model = joblib.load('models/marginpredictor.pkl')

    # Step 4: Predict win probabilities
    print("Predicting win probabilities...")
    win_estimator = WinProbabilityEstimator(data=None)
    win_estimator.model = win_model
    win_estimator.scaler = win_scaler
    predicted_win_probabilities = win_estimator.predict_win_probability(data_test_clean)

    # Step 5: Predict margins
    print("Predicting margins...")
    margin_estimator = MarginPredictor(data=None)
    margin_estimator.model = margin_model
    predicted_margin = margin_estimator.model.predict(data_test_clean)

    # Step 6: Predict team points
    print("Predicting points...")
    points_estimator = PointsPredictor(data=None)
    points_estimator.model = points_predictor_model
    predicted_points = points_estimator.model.predict(data_test_clean)

    # Step 7: Combine results into a DataFrame
    print("Creating predictions DataFrame...")
    df_predictions = create_predictions_df(
        list_home,
        list_away,
        predicted_points,
        predicted_margin,
        predicted_win_probabilities
    )

    # Step 8: Output predictions
    print("Predictions DataFrame:")
    print(df_predictions)

if __name__ == "__main__":
    main()