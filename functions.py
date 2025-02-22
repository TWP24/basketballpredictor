import http.client
import json
import csv
import pandas as pd
import joblib
def fetch_and_save_basketball_standings(api_host, api_key, league, season):

    # Set up the API connection
    conn = http.client.HTTPSConnection(api_host)
    headers = {
        'x-rapidapi-host': api_host,
        'x-rapidapi-key': api_key
    }
    
    # Construct the API endpoint with the given league and season
    endpoint = f"/standings?league={league}&season={season}"
    conn.request("GET", endpoint, headers=headers)

    # Get the API response
    res = conn.getresponse()
    data = res.read()

    # Parse the JSON response
    response_json = json.loads(data.decode("utf-8"))

    # Define the output file name
    output_file = f"nba_standings_{league}_{season.replace('-', '_')}.csv"

    # Write the data to the CSV file
    with open(output_file, mode="w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)

        # Write the header row
        writer.writerow([
            "Position", "Team Name", "Played Games", "Wins", "Win Percentage",
            "Losses", "Loss Percentage", "Points For", "Points Against",
            "Form", "Description"
        ])

        # Extract and write data for each team
        if response_json.get("results", 0) > 0:
            standings = response_json["response"][0]  # The first group of standings
            for team in standings:
                writer.writerow([
                    team["position"],
                    team["team"]["name"],
                    team["games"]["played"],
                    team["games"]["win"]["total"],
                    team["games"]["win"]["percentage"],
                    team["games"]["lose"]["total"],
                    team["games"]["lose"]["percentage"],
                    team["points"]["for"],
                    team["points"]["against"],
                    team["form"],
                    team["description"]
                ])

    print(f"Data successfully written to '{output_file}'")
    return output_file  


###############################################################################################################
###############################################################################################################
###############################################################################################################




def fetch_and_save_basketball_games(api_host, api_key, league, season):
    """
    Fetch basketball game data from the API and save it to a CSV file.
    If the resulting CSV file has 0 rows, retry with a modified season format.
    """

    # Set up the API connection
    conn = http.client.HTTPSConnection(api_host)
    headers = {
        'x-rapidapi-host': api_host,
        'x-rapidapi-key': api_key
    }

    # Construct the API endpoint
    endpoint = f"/games?league={league}&season={season}"
    conn.request("GET", endpoint, headers=headers)

    # Get the API response
    res = conn.getresponse()
    data = res.read()

    # Decode the JSON response
    decoded_data = json.loads(data.decode("utf-8"))

    # Define the output file name
    output_file = f"basketball_games_{league}_{season.replace('-', '_')}.csv"

    # Write the data to the CSV file
    with open(output_file, mode='w', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)

        # Write the header row with more variables
        writer.writerow([
            'Game ID', 'Date', 'Time', 'League Name', 'Season', 'Country',
            'Home Team', 'Away Team', 'Home Logo', 'Away Logo',
            'Home Q1 Score', 'Home Q2 Score', 'Home Q3 Score', 'Home Q4 Score', 'Home Total Score',
            'Away Q1 Score', 'Away Q2 Score', 'Away Q3 Score', 'Away Q4 Score', 'Away Total Score',
            'Game Status', 'Timezone'
        ])

        # Loop through the games and write rows
        for game in decoded_data.get('response', []):
            writer.writerow([
                game.get('id'),                              # Game ID
                game.get('date'),                            # Date
                game.get('time'),                            # Time
                game.get('league', {}).get('name'),          # League Name
                game.get('league', {}).get('season'),        # Season
                game.get('country', {}).get('name'),         # Country Name
                game['teams']['home']['name'],               # Home Team Name
                game['teams']['away']['name'],               # Away Team Name
                game['teams']['home']['logo'],               # Home Team Logo
                game['teams']['away']['logo'],               # Away Team Logo
                game['scores']['home']['quarter_1'],         # Home Q1 Score
                game['scores']['home']['quarter_2'],         # Home Q2 Score
                game['scores']['home']['quarter_3'],         # Home Q3 Score
                game['scores']['home']['quarter_4'],         # Home Q4 Score
                game['scores']['home']['total'],             # Home Total Score
                game['scores']['away']['quarter_1'],         # Away Q1 Score
                game['scores']['away']['quarter_2'],         # Away Q2 Score
                game['scores']['away']['quarter_3'],         # Away Q3 Score
                game['scores']['away']['quarter_4'],         # Away Q4 Score
                game['scores']['away']['total'],             # Away Total Score
                game['status']['long'],                      # Game Status (e.g., "Game Finished")
                game.get('timezone')                         # Timezone
            ])

    # Check if the CSV file has 0 rows
    df = pd.read_csv(output_file)
    if df.shape[0] == 0:  # If no rows exist in the CSV file
        modified_season = "2024"
        return fetch_and_save_basketball_games(api_host, api_key, league, modified_season)

    print(f"Detailed data has been written to {output_file}")
    return output_file  # Return the path to the CSV file



###############################################################################################################
###############################################################################################################
###############################################################################################################




def merge_basketball_data(games_csv, standings_csv):
    # Load the CSV files
    games_df = pd.read_csv(games_csv)
    standings_df = pd.read_csv(standings_csv)
    # Perform the merge for "Home Team"
    merged_df = pd.merge(
        games_df,
        standings_df,
        left_on='Home Team',
        right_on='Team Name',
        how='left',
        suffixes=('', '_Home')
    )

    # Rename the relevant columns for Home Team
    home_columns = ['Wins', 'Win Percentage', 'Losses', 'Loss Percentage', 'Points For', 'Points Against']
    for col in home_columns:
        merged_df.rename(columns={col: f"Home_{col}"}, inplace=True)

    # Drop the extra "Team Name" column from the Home Team merge
    merged_df.drop(columns=['Team Name'], inplace=True)

    # Perform the merge for "Away Team"
    merged_df = pd.merge(
        merged_df,
        standings_df,
        left_on='Away Team',
        right_on='Team Name',
        how='left',
        suffixes=('', '_Away')
    )

    # Rename the relevant columns for Away Team
    for col in home_columns:
        merged_df.rename(columns={col: f"Away_{col}"}, inplace=True)

    # Drop the extra "Team Name" column from the Away Team merge
    merged_df.drop(columns=['Team Name'], inplace=True)

    # Return the merged DataFrame
    return merged_df



###############################################################################################################
###############################################################################################################
###############################################################################################################
def fetch_and_merge_basketball_data(api_host, api_key, league, season):
    # Fetch and save the games and standings data
    games_csv = fetch_and_save_basketball_games(api_host, api_key, league, season)

    # Fetch, process, and save the statistics data
    df = fetch_statistics(league, season, api_host, api_key)
    data = extract_standings(df)  # Process the standings data
    stats_csv = "stats.csv"  # Define the file name for the stats CSV
    data.to_csv(stats_csv, index=False)  # Save the DataFrame to the CSV file

    # Merge the data using the existing merge_basketball_data function
    merged_data = merge_basketball_data(games_csv, stats_csv)

    # Return the merged DataFrame
    return merged_data

import pandas as pd

###############################################################################################################
###############################################################################################################
###############################################################################################################

def prepare_prediction_data(league, api_host, api_key, fetch_and_merge_basketball_data):
    # Define the possible season formats
     # Cover all naming conventions
    season = "2024-2025" 
    season_data = fetch_and_merge_basketball_data(api_host, api_key, league, season)
    # Initialize an empty DataFrame
    testdata = season_data.copy()
    testdata.to_csv('testdata_for_prediction.csv', index=False)
    # Filter matches that are "Not Started"
    testdata_for_prediction = testdata[testdata['Game Status'] == 'Not Started']

    # Return the filtered DataFrame
    return testdata_for_prediction
###############################################################################################################
###############################################################################################################
###############################################################################################################

import pandas as pd

def process_data_main(input_data):

    # Extract the 'Home Team' and 'Away Team' values into separate lists
    list_home = input_data['Home Team'].tolist()
    list_away = input_data['Away Team'].tolist()

    # Remove 'Home Team' and 'Away Team' columns from the input data
    data_without_teams = input_data.drop(columns=['Home Team', 'Away Team'])

    return data_without_teams, list_home, list_away

###############################################################################################################
###############################################################################################################
###############################################################################################################

def filter_relevant_columns_main(data):
    relevant_columns = [
        'Home_Wins', 'Home_Win Percentage', 'Home_Losses', 'Home_Loss Percentage',
        'Home_Points For', 'Home_Points Against',
        'Away_Wins', 'Away_Win Percentage', 'Away_Losses', 'Away_Loss Percentage',
        'Away_Points For', 'Away_Points Against'
    ]

    # Filter only the relevant columns
    filtered_data = data[relevant_columns]

    # Drop rows with NaN values


    return filtered_data

###############################################################################################################
###############################################################################################################
###############################################################################################################

def create_predictions_df(list_home, list_away, predicted_points, predicted_margin, predicted_win_probabilities):

    
    # Extract the predicted home and away scores (assuming predicted_points has two columns)
    home_scores = predicted_points[:, 0]  # Home scores
    away_scores = predicted_points[:, 1]  # Away scores

    # Create a dictionary for the DataFrame
    data = {
        'Home_team': list_home,
        'Away_team': list_away,
        'Home_score': home_scores,
        'Away_score': away_scores,
        'Margin_of_Victory': predicted_margin,
        'Probability_Winning_Home': predicted_win_probabilities
    }

    # Create the DataFrame
    df_predictions = pd.DataFrame(data)

    return df_predictions
###############################################################################################################
###############################################################################################################
###############################################################################################################

def get_team_ids(league, season, api_host, api_key):
    # Helper function to make the API request
    def fetch_ids(league, season):
        conn = http.client.HTTPSConnection(api_host)
        headers = {
            'x-rapidapi-host': api_host,
            'x-rapidapi-key': api_key
        }
        endpoint = f"/teams?league={league}&season={season}"
        conn.request("GET", endpoint, headers=headers)

        res = conn.getresponse()
        data = res.read()
        response_data = json.loads(data.decode("utf-8"))

        return [team["id"] for team in response_data.get("response", [])]

    # Try the given season format
    ids_list = fetch_ids(league, season)

    # If no results, try the alternate season format
    if not ids_list:
        # Determine the alternate season format
        if "-" in season:
            alternate_season = season.split("-")[0]
        else:
            alternate_season = f"{season}-{int(season)+1}"

        ids_list = fetch_ids(league, alternate_season)

    return ids_list

###############################################################################################################
###############################################################################################################
###############################################################################################################
def fetch_statistics(league, season, api_host, api_key):
    """
    Fetch statistics for all teams in a given league and season, handling dynamic season formats.
    Tries both season formats (e.g., "2024" and "2024-2025") if needed.
    """
    conn = http.client.HTTPSConnection(api_host)
    headers = {
        'x-rapidapi-host': api_host,
        'x-rapidapi-key': api_key
    }

    # Fetch all team IDs for the league and season using the get_team_ids function
    team_ids = get_team_ids(league, season, api_host, api_key)
    data_list = []  # List to store rows for the DataFrame

    for team_id in team_ids:
        # Try fetching statistics with the given season format
        endpoint = f"/statistics?league={league}&team={team_id}&season={season}"
        conn.request("GET", endpoint, headers=headers)
        res = conn.getresponse()
        data = res.read()
        parsed_data = json.loads(data.decode("utf-8"))

        if "response" not in parsed_data:
            # If the response is empty, try fetching with the alternative season format
            alternative_season = f"{season}-{int(season.split('-')[0]) + 1}" if "-" not in season else season.split('-')[0]
            endpoint = f"/statistics?league={league}&team={team_id}&season={alternative_season}"
            conn.request("GET", endpoint, headers=headers)
            res = conn.getresponse()
            data = res.read()
            parsed_data = json.loads(data.decode("utf-8"))

        # Extract the relevant statistics from the response
        response = parsed_data.get('response', {})
        country = response.get('country', {})
        league_data = response.get('league', {})
        team = response.get('team', {})
        games = response.get('games', {})
        points = response.get('points', {})

        # Flatten the data into a dictionary
        row = {
            'Country ID': country.get('id'),
            'Country Name': country.get('name'),
            'Country Code': country.get('code'),
            'Country Flag URL': country.get('flag'),
            'League ID': league_data.get('id'),
            'League Name': league_data.get('name'),
            'League Type': league_data.get('type'),
            'League Season': league_data.get('season'),
            'League Logo URL': league_data.get('logo'),
            'Team ID': team.get('id'),
            'Team Name': team.get('name'),
            'Team Logo URL': team.get('logo'),
            'Games Played (Home)': games.get('played', {}).get('home'),
            'Games Played (Away)': games.get('played', {}).get('away'),
            'Games Played (All)': games.get('played', {}).get('all'),
            'Wins (Home)': games.get('wins', {}).get('home', {}).get('total'),
            'Wins Percentage (Home)': games.get('wins', {}).get('home', {}).get('percentage'),
            'Wins (Away)': games.get('wins', {}).get('away', {}).get('total'),
            'Wins Percentage (Away)': games.get('wins', {}).get('away', {}).get('percentage'),
            'Wins (All)': games.get('wins', {}).get('all', {}).get('total'),
            'Wins Percentage (All)': games.get('wins', {}).get('all', {}).get('percentage'),
            'Losses (Home)': games.get('loses', {}).get('home', {}).get('total'),
            'Losses Percentage (Home)': games.get('loses', {}).get('home', {}).get('percentage'),
            'Losses (Away)': games.get('loses', {}).get('away', {}).get('total'),
            'Losses Percentage (Away)': games.get('loses', {}).get('away', {}).get('percentage'),
            'Losses (All)': games.get('loses', {}).get('all', {}).get('total'),
            'Losses Percentage (All)': games.get('loses', {}).get('all', {}).get('percentage'),
            'Points For (Home)': points.get('for', {}).get('total', {}).get('home'),
            'Points For (Away)': points.get('for', {}).get('total', {}).get('away'),
            'Points For (All)': points.get('for', {}).get('total', {}).get('all'),
            'Points For Average (Home)': points.get('for', {}).get('average', {}).get('home'),
            'Points For Average (Away)': points.get('for', {}).get('average', {}).get('away'),
            'Points For Average (All)': points.get('for', {}).get('average', {}).get('all'),
            'Points Against (Home)': points.get('against', {}).get('total', {}).get('home'),
            'Points Against (Away)': points.get('against', {}).get('total', {}).get('away'),
            'Points Against (All)': points.get('against', {}).get('total', {}).get('all'),
            'Points Against Average (Home)': points.get('against', {}).get('average', {}).get('home'),
            'Points Against Average (Away)': points.get('against', {}).get('average', {}).get('away'),
            'Points Against Average (All)': points.get('against', {}).get('average', {}).get('all'),
        }

        data_list.append(row)

    # Create the DataFrame from the collected data
    df = pd.DataFrame(data_list)

    # Check for 'None' values in the 'Country ID' column
    if df['Country ID'].isnull().any():
        # If 'None' values are found, retry with a modified season format
        modified_season = season.split('-')[0] if "-" in season else f"{season}-2025"  # Adjust as necessary
        print(f"Retrying with modified season format: {modified_season}")
        return fetch_statistics(league, modified_season, api_host, api_key)

    return df

###############################################################################################################
###############################################################################################################
###############################################################################################################
def extract_standings(df):
    # Mapping input DataFrame variables to the desired output format
    standings_df = pd.DataFrame({
        "Position": 0,  # Setting Position to 0 for all rows
        "Team Name": df["Team Name"],
        "Played Games": df["Games Played (All)"],
        "Wins": df["Wins (All)"],
        "Win Percentage": df["Wins Percentage (All)"],
        "Losses": df["Losses (All)"],
        "Loss Percentage": df["Losses Percentage (All)"],
        "Points For": df["Points For (All)"],
        "Points Against": df["Points Against (All)"],
        "Form": None,  # Placeholder for "Form" as it's not in the input DataFrame
        "Description": None  # Placeholder for "Description" as it's not in the input DataFrame
    })

    return standings_df

###############################################################################################################
###############################################################################################################
###############################################################################################################
def replace_nan_with_value(df, value=0.5):
    return df.fillna(value)
###############################################################################################################
###############################################################################################################
###############################################################################################################


