
# Basketball Game Predictions App

This is a Streamlit-based application for predicting basketball game outcomes, including win probabilities, score margins, and team points. Follow the steps below to set up and use the application.

---

## Features

- Predict **win probabilities** for upcoming basketball games.
- Estimate the **margin of victory** and **team points**.
- Display predictions in an interactive table.

---

## Requirements

### Software
- Python 3.8 or above
- Streamlit
- Joblib

### Data
- Pre-trained models for predictions:
  - `points_predictor_model.pkl`
  - `win_model.pkl`
  - `win_scaler.pkl`
  - `marginpredictor.pkl`

---

## Installation

1. **Clone the Repository**
   ```bash
   git clone <repository_url>
   cd <repository_folder>
   ```

2. **Install Dependencies**
   Use a virtual environment to avoid dependency conflicts:
   ```bash
   python -m venv env
   source env/bin/activate   # On Windows: env\Scripts\activate
   pip install -r requirements.txt
   ```

3. **Ensure Model Files Exist**
   Place the following files in a `models/` folder within the project directory:
   - `points_predictor_model.pkl`
   - `win_model.pkl`
   - `win_scaler.pkl`
   - `marginpredictor.pkl`

---

## Usage

1. **Run the App**
   ```bash
   streamlit run app.py
   ```

2. **Access the Application**
   - Open a web browser and go to: [http://localhost:8501](http://localhost:8501).

3. **Input Details**
   - Enter the **League ID** of the basketball league.
   - Provide your **API Key** (the application will prompt you).

4. **Generate Predictions**
   - Click the **"Generate Predictions"** button to view results.
   - The app will fetch match data, process it, and display predictions.

---

## API Key Information

The app uses the API from `v1.basketball.api-sports.io`. You must:
- Register at [API Sports](https://www.api-sports.io/).
- Obtain an API Key.
- Enter your key securely when prompted in the app.

---

## Troubleshooting

- **No Data Found**: Ensure the League ID is valid.
- **Error Messages**: Check that the model files are correctly placed in the `models/` folder.
- **Port in Use**: If the app fails to start, try:
  ```bash
  streamlit run app.py --server.port=<new_port_number>
  ```

---

## Example Usage

1. **Enter League ID**: Example: `1234`
2. **Enter API Key**: Example: `abcd1234`
3. View predictions for all upcoming matches in the league.

---

## Additional Notes

- Predictions are for informational purposes and depend on the accuracy of input data.
- For further customization or support, contact the development team.
