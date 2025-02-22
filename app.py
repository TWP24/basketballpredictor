import streamlit as st
import os
import joblib
from classes import WinProbabilityEstimator, PointsPredictor, MarginPredictor
from functions import (
    prepare_prediction_data,
    fetch_and_merge_basketball_data,
    process_data_main,
    filter_relevant_columns_main,
    create_predictions_df,
    replace_nan_with_value
)

# Load models and scalers (updated caching methods)
base_path = os.path.dirname(__file__)  # Get the directory of the current script
models_path = os.path.join(base_path, 'models')
points_predictor_model = joblib.load(os.path.join(models_path, 'points_predictor_model.pkl'))
win_model = joblib.load(os.path.join(models_path, 'win_model (1).pkl'))
win_scaler = joblib.load(os.path.join(models_path, 'win_scaler.pkl'))
margin_model = joblib.load(os.path.join(models_path, 'marginpredictor.pkl'))

def main():
    st.title("Basketball Game Predictions")

    league = st.text_input("Enter the League ID:")
    api_key = st.text_input("Enter your API Key:", type="password")  # Securely input API Key

    if st.button("Generate Predictions"):
        if league and api_key:
            st.write("Fetching all matches for the league...")

            # Step 1: Fetch matches data
            testdata_for_prediction = prepare_prediction_data(
                league, 
                "v1.basketball.api-sports.io", 
                api_key, 
                fetch_and_merge_basketball_data
            )

            if testdata_for_prediction is None or len(testdata_for_prediction) == 0:
                st.error("No data found for the given league. Please check the League ID.")
                return

            st.write(f"Displaying all matches for league {league}:")
            st.dataframe(testdata_for_prediction)

            # Step 2: Process data for prediction
            st.write("Processing data for prediction...")
            data_test = testdata_for_prediction.copy()
            data_test_no_teams, list_home, list_away = process_data_main(data_test)
            data_test_clean = filter_relevant_columns_main(data_test_no_teams)
            data_test_clean = replace_nan_with_value(data_test_clean, 0.5)  # Replace NaN values

            # Step 3: Predict Win Probabilities
            st.write("Predicting win probabilities...")
            win_estimator = WinProbabilityEstimator(data=None)
            win_estimator.model = win_model
            win_estimator.scaler = win_scaler
            predicted_win_probabilities = win_estimator.predict_win_probability(data_test_clean)

            # Step 4: Predict margins
            st.write("Predicting margins...")
            margin_estimator = MarginPredictor(data=None)
            margin_estimator.model = margin_model
            predicted_margin = margin_estimator.model.predict(data_test_clean)

            # Step 5: Predict team points
            st.write("Predicting points...")
            points_estimator = PointsPredictor(data=None)
            points_estimator.model = points_predictor_model
            predicted_points = points_estimator.model.predict(data_test_clean)

            # Step 6: Combine the results into a DataFrame
            st.write("Creating predictions DataFrame...")
            df_predictions = create_predictions_df(
                list_home,
                list_away,
                predicted_points,
                predicted_margin,
                predicted_win_probabilities
            )

            # Step 7: Show the results in a table
            st.write("Predictions for all matches:")
            st.dataframe(df_predictions)

        else:
            st.error("Please provide both the League ID and your API Key.")

if __name__ == "__main__":
    main()