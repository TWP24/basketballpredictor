import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report
from sklearn.preprocessing import StandardScaler

class WinProbabilityEstimator:
    def __init__(self, data):
        # Initialize with the dataset
        self.data = data
        self.model = None
        self.X = None
        self.y = None
        self.scaler = StandardScaler()

    def preprocess_data(self):
        # Drop 'Win' and 'Margin' for feature selection
        self.X = self.data.drop(columns=['Win', 'Margin'])
        self.y = self.data['Win']
        
        # Feature scaling (important for models like Logistic Regression)
        self.X = self.scaler.fit_transform(self.X)

    def train_model(self):
        # Create the logistic regression model with L2 regularization (Ridge)
        self.model = LogisticRegression(solver='liblinear', C=1.0, penalty='l2')
        
        # Cross-validation to evaluate the model
        cv_scores = cross_val_score(self.model, self.X, self.y, cv=5, scoring='accuracy')
        print(f"Cross-validation accuracy scores: {cv_scores}")
        print(f"Mean cross-validation accuracy: {cv_scores.mean():.2f}")
        
        # Train the model on the full dataset
        self.model.fit(self.X, self.y)
    
    def evaluate_model(self):
        # Evaluate the model's performance on the training data
        y_pred = self.model.predict(self.X)
        accuracy = accuracy_score(self.y, y_pred)
        roc_auc = roc_auc_score(self.y, self.model.predict_proba(self.X)[:, 1])
        
        print(f"Model Evaluation:")
        print(f"Accuracy: {accuracy:.2f}")
        print(f"ROC AUC Score: {roc_auc:.2f}")
        print(f"Classification Report:\n{classification_report(self.y, y_pred)}")
    
    def predict_win_probability(self, input_data):
        # Predict win probability (1 = Home Win, 0 = Away Win)
        input_data_scaled = self.scaler.transform(input_data)  # Don't forget to scale input data
        return self.model.predict_proba(input_data_scaled)[:, 1]
    
###############################################################################################################
###############################################################################################################
###############################################################################################################

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.multioutput import MultiOutputRegressor
from sklearn.metrics import mean_squared_error, r2_score
import pandas as pd
import numpy as np

class PointsPredictor:
    def __init__(self, data):
        """
        Initializes the PointsPredictor class.
        
        Parameters:
            data (pd.DataFrame): The dataset containing features and the target variables ('Home Total Score' and 'Away Total Score').
        """
        self.data = data
        self.model = None

    def preprocess_data(self):
        """
        Splits the dataset into features (X) and target variables (y), and performs a train-test split.
        """
        # Separate features (exclude 'Home Total Score' and 'Away Total Score') and target ('Home Total Score' and 'Away Total Score')
        self.X = self.data.drop(columns=['Home Total Score', 'Away Total Score'])
        self.y = self.data[['Home Total Score', 'Away Total Score']]

        # Train-test split (80-20 split)
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X, self.y, test_size=0.2, random_state=42
        )

    def train_model(self, fine_tune=True):
        """
        Trains the model to predict both 'Home Total Score' and 'Away Total Score', with optional fine-tuning.

        Parameters:
            fine_tune (bool): Whether to perform hyperparameter optimization. Default is True.
        """
        if fine_tune:
            # Define hyperparameter grid for Random Forest
            param_grid = {
                'n_estimators': [50, 100, 200],
                'max_depth': [None, 10, 20, 30],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4]
            }

            # Perform Grid Search Cross-Validation
            grid_search = GridSearchCV(
                RandomForestRegressor(random_state=42),
                param_grid,
                cv=3,
                scoring='neg_mean_squared_error',
                n_jobs=-1,
                verbose=2
            )
            grid_search.fit(self.X_train, self.y_train)

            # Use the best model from the grid search to fit MultiOutputRegressor
            self.model = MultiOutputRegressor(grid_search.best_estimator_)
            self.model.fit(self.X_train, self.y_train)  # Fit the model here

            print("Best Hyperparameters:", grid_search.best_params_)

        else:
            # Train a MultiOutputRegressor with default parameters
            self.model = MultiOutputRegressor(RandomForestRegressor(random_state=42))
            self.model.fit(self.X_train, self.y_train)  # Fit the model here

        # Evaluate the model
        self.evaluate_model()

    def evaluate_model(self):
        """
        Evaluates the trained model on the test set and prints performance metrics.
        """
        if self.model is None:
            raise ValueError("Model has not been trained yet.")

        # Make predictions on the test set
        y_pred = self.model.predict(self.X_test)

        # Calculate evaluation metrics for both Home Total Score and Away Total Score
        mse = mean_squared_error(self.y_test, y_pred)
        r2 = r2_score(self.y_test, y_pred)

        print(f"Model Evaluation:\nMean Squared Error: {mse:.2f}\nR^2 Score: {r2:.2f}")

    def predict(self, input_data):
        """
        Predicts the 'Home Total Score' and 'Away Total Score' for the given input data.

        Parameters:
            input_data (pd.DataFrame): The input data excluding the target variables ('Home Total Score' and 'Away Total Score').

        Returns:
            np.ndarray: The predicted values for both 'Home Total Score' and 'Away Total Score'.
        """
        if self.model is None:
            raise ValueError("Model has not been trained yet.")

        # Ensure input_data has the same columns as the training features
        if set(input_data.columns) != set(self.X.columns):
            raise ValueError("Input data columns do not match training data columns.")

        return self.model.predict(input_data)



###############################################################################################################
###############################################################################################################
###############################################################################################################


from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_squared_error, r2_score
import pandas as pd
import numpy as np

class MarginPredictor:
    def __init__(self, data):
        """
        Initializes the MarginPredictor class.
        
        Parameters:
            data (pd.DataFrame): The dataset containing features and the target variable 'Margin'.
        """
        self.data = data
        self.model = None

    def preprocess_data(self):
        """
        Splits the dataset into features (X) and target (y), and performs a train-test split.
        """
        # Separate features (exclude 'Win') and target ('Margin')
        self.X = self.data.drop(columns=['Margin', 'Win'])
        self.y = self.data['Margin']

        # Train-test split (80-20 split)
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X, self.y, test_size=0.2, random_state=42
        )

    def train_model(self, fine_tune=True):
        """
        Trains the Random Forest Regressor model, with optional fine-tuning.

        Parameters:
            fine_tune (bool): Whether to perform hyperparameter optimization. Default is True.
        """
        if fine_tune:
            # Define hyperparameter grid for Random Forest
            param_grid = {
                'n_estimators': [50, 100, 200],
                'max_depth': [None, 10, 20, 30],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4]
            }

            # Perform Grid Search Cross-Validation
            grid_search = GridSearchCV(
                RandomForestRegressor(random_state=42),
                param_grid,
                cv=3,
                scoring='neg_mean_squared_error',
                n_jobs=-1,
                verbose=2
            )
            grid_search.fit(self.X_train, self.y_train)

            # Use the best model
            self.model = grid_search.best_estimator_
            print("Best Hyperparameters:", grid_search.best_params_)

        else:
            # Train a Random Forest Regressor with default parameters
            self.model = RandomForestRegressor(random_state=42)
            self.model.fit(self.X_train, self.y_train)

        # Evaluate the model
        self.evaluate_model()

    def evaluate_model(self):
        """
        Evaluates the trained model on the test set and prints performance metrics.
        """
        if self.model is None:
            raise ValueError("Model has not been trained yet.")

        # Make predictions on the test set
        y_pred = self.model.predict(self.X_test)

        # Calculate evaluation metrics
        mse = mean_squared_error(self.y_test, y_pred)
        r2 = r2_score(self.y_test, y_pred)

        print(f"Model Evaluation:\nMean Squared Error: {mse:.2f}\nR^2 Score: {r2:.2f}")

    def predict(self, input_data):
        """
        Predicts the 'Margin' for the given input data.

        Parameters:
            input_data (pd.DataFrame): The input data excluding 'Win'.

        Returns:
            np.ndarray: The predicted 'Margin' values.
        """
        if self.model is None:
            raise ValueError("Model has not been trained yet.")

        # Ensure input_data has the same columns as the training features
        if set(input_data.columns) != set(self.X.columns):
            raise ValueError("Input data columns do not match training data columns.")

        return self.model.predict(input_data)
