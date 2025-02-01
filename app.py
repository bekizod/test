import pandas as pd
import sys
import uvicorn
import pickle
from fastapi import FastAPI, HTTPException
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import os
from fastapi.middleware.cors import CORSMiddleware

# Global preprocessor and model definition
preprocessor = None
model = None

# Step 1: Data Preprocessing
def preprocess_data(df):
    global preprocessor
    # Drop unnecessary columns
    # Drop unnecessary columns
    df = df.drop(columns=[col for col in [
        'timestamp', 'date_GMT', 'home_team_shots', 'over_35_percentage_pre_match', 
        'away_ppg', 'home_team_goal_count_half_time', 'over_25_percentage_pre_match', 
        'status', 'over_05_HT_FHG_percentage_pre_match', 'odds_ft_home_team_win', 
        'home_team_corner_count', 'odds_ft_over25', 'Away Team Pre-Match xG', 
        'Pre-Match PPG (Away)', 'away_team_first_half_cards', 'attendance.1', 
        'home_team_goal_timings', 'over_15_2HG_percentage_pre_match', 
        'odds_ft_away_team_win', 'home_ppg', 'Game Week', 'away_team_goal_timings', 
        'away_team_second_half_cards', 'Pre-Match PPG (Home)', 'attendance', 
        'odds_ft_draw', 'away_team_goal_count_half_time', 'btts_percentage_pre_match', 
        'odds_btts_yes', 'stadium_name', 'team_a_xg', 'home_team_first_half_cards', 
        'home_team_shots_off_target', 'Home Team Pre-Match xG', 'odds_ft_over35', 
        'odds_ft_over45', 'referee', 'home_team_second_half_cards', 
        'over_45_percentage_pre_match', 'away_team_name', 'away_team_corner_count', 
        'away_team_shots', 'over_05_2HG_percentage_pre_match', 
        'over_15_percentage_pre_match', 'average_cards_per_match_pre_match', 
        'average_corners_per_match_pre_match', 'odds_ft_over15', 
        'average_goals_per_match_pre_match', 'away_team_shots_off_target', 
        'total_goals_at_half_time', 'odds_btts_no', 'total_goal_count', 'team_b_xg', 
        'over_15_HT_FHG_percentage_pre_match','possession_difference', 'total_shots_on_target',
    ] if col in df.columns])

    # Creating new features
    df['possession_difference'] = df['home_team_possession'] - df['away_team_possession']
    df['total_shots_on_target'] = df['home_team_shots_on_target'] + df['away_team_shots_on_target']

    # Handle missing values in numeric columns
    numeric_columns = df.select_dtypes(include=['number']).columns
    df[numeric_columns] = df[numeric_columns].fillna(0)

    # Convert all numeric columns to float (to handle mixed types)
    df[numeric_columns] = df[numeric_columns].astype(float)

    # Encoding match results as labels
    df['match_result'] = df.apply(lambda row: 1 if row['home_team_goal_count'] > row['away_team_goal_count']
                                  else (-1 if row['home_team_goal_count'] < row['away_team_goal_count'] else 0), axis=1)

    # Select numerical and categorical features
    numerical_features = df.select_dtypes(include=['number']).columns.tolist()
    numerical_features.remove('match_result')  # Exclude target variable

    categorical_features = df.select_dtypes(include=['object']).columns.tolist()

    # Initialize the preprocessor here
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numerical_features),
            ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
        ])
    
    return df, numerical_features, categorical_features


# Step 2: Model Training
def train_model(X_train, y_train):
    model = RandomForestClassifier(random_state=42)

    param_grid = {
        'n_estimators': [50, 100, 200],
        'max_depth': [None, 10, 20, 30],
        'min_samples_split': [2, 5, 10]
    }

    grid_search = GridSearchCV(model, param_grid, cv=5, scoring='accuracy')
    grid_search.fit(X_train, y_train)

    best_model = grid_search.best_estimator_
    print("Best Parameters:", grid_search.best_params_)
    print("Best Accuracy:", grid_search.best_score_)

    return best_model

# Step 3: Model Evaluation
def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)

    print("Classification Report:\n", classification_report(y_test, y_pred))
    print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
    print("Model Accuracy:", accuracy_score(y_test, y_pred))

# Step 4: Save Model and Preprocessor
def save_model_and_preprocessor(model, preprocessor, model_filename="model.pkl", preprocessor_filename="preprocessor.pkl"):
    with open(model_filename, "wb") as f:
        pickle.dump(model, f)
    with open(preprocessor_filename, "wb") as f:
        pickle.dump(preprocessor, f)
    print(f"Model saved to {model_filename}")
    print(f"Preprocessor saved to {preprocessor_filename}")

# Step 5: FastAPI App
app = FastAPI()


app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],  # Add your frontend URL here
    allow_credentials=True,
    allow_methods=["*"],  # Allow all HTTP methods (GET, POST, etc.)
    allow_headers=["*"],  # Allow all headers
)

# Load trained model and preprocessor
if os.path.exists("model.pkl") and os.path.exists("preprocessor.pkl"):
    with open("model.pkl", "rb") as f:
        model = pickle.load(f)
    with open("preprocessor.pkl", "rb") as f:
        preprocessor = pickle.load(f)
    print("Model and preprocessor loaded successfully!")
else:
    print("Model or preprocessor not found! Please train the model first.")
    model = None
    preprocessor = None

@app.post("/predict/")
async def predict(data: dict):
    if model is None:
        raise HTTPException(status_code=500, detail="Model not loaded. Train the model first.")
    if preprocessor is None:
        raise HTTPException(status_code=500, detail="Preprocessor not loaded. Train the model first.")

    try:
        # Preprocess input data as per training preprocessing
        df = pd.DataFrame([data])

        # Ensure numeric columns are properly formatted
        numeric_columns = df.select_dtypes(include=['number']).columns
        df[numeric_columns] = df[numeric_columns].fillna(0).astype(float)

        # Apply preprocessor transformations (same as during training)
        processed_data = preprocessor.transform(df)

        # Predict using the trained model
        prediction = model.predict(processed_data)
        return {"prediction": int(prediction[0])}
    
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/predict_match/")
async def predict_match(data: dict):
    if model is None or preprocessor is None:
        raise HTTPException(status_code=500, detail="Model or preprocessor not loaded. Train the model first.")

    try:
        # Ensure both teams' data is provided
        if "home_team" not in data or "away_team" not in data:
            raise HTTPException(status_code=400, detail="Both home_team and away_team data are required.")

        # Convert home and away team data into DataFrames
        home_team_df = pd.DataFrame([data["home_team"]])
        away_team_df = pd.DataFrame([data["away_team"]])

        # Ensure all required columns exist
        required_columns = {
            "home_team_possession", "home_team_shots_on_target", "home_team_yellow_cards",
            "home_team_red_cards", "home_team_goal_count", "home_team_fouls",
            "away_team_yellow_cards", "away_team_red_cards", "away_team_goal_count", 
            "away_team_fouls"
        }

        # Check if required columns are missing
        missing_columns = required_columns - set(home_team_df.columns) - set(away_team_df.columns)
        if missing_columns:
            raise HTTPException(status_code=400, detail=f"Columns are missing: {missing_columns}")

        # Ensure numeric columns are properly formatted
        numeric_columns_home = home_team_df.select_dtypes(include=['number']).columns
        numeric_columns_away = away_team_df.select_dtypes(include=['number']).columns

        home_team_df[numeric_columns_home] = home_team_df[numeric_columns_home].fillna(0).astype(float)
        away_team_df[numeric_columns_away] = away_team_df[numeric_columns_away].fillna(0).astype(float)

        # Merge both teams' data for prediction
        match_df = home_team_df.copy()
        match_df['away_team_possession'] = away_team_df['home_team_possession']
        match_df['away_team_shots_on_target'] = away_team_df['home_team_shots_on_target']
        match_df['away_team_yellow_cards'] = away_team_df['away_team_yellow_cards']
        match_df['away_team_red_cards'] = away_team_df['away_team_red_cards']
        match_df['away_team_goal_count'] = away_team_df['away_team_goal_count']
        match_df['away_team_fouls'] = away_team_df['away_team_fouls']

        match_df['possession_difference'] = match_df['home_team_possession'] - match_df['away_team_possession']
        match_df['total_shots_on_target'] = match_df['home_team_shots_on_target'] + match_df['away_team_shots_on_target']
        match_df['total_fouls'] = match_df['home_team_fouls'] + match_df['away_team_fouls']
        match_df['total_yellow_cards'] = match_df['home_team_yellow_cards'] + match_df['away_team_yellow_cards']
        match_df['total_red_cards'] = match_df['home_team_red_cards'] + match_df['away_team_red_cards']

        # Apply preprocessor transformations
        processed_data = preprocessor.transform(match_df)

        # Predict match result
        prediction = model.predict(processed_data)
        result = "Home Team Wins" if prediction[0] == 1 else ("Away Team Wins" if prediction[0] == -1 else "Draw")

        return {"prediction": result}

    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))



# Main function to run training and FastAPI
def main():
    df = pd.read_csv("football.csv")  # Ensure your dataset exists

    df, num_features, cat_features = preprocess_data(df)
    X = df.drop(columns=['match_result'])
    y = df['match_result']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    # Apply transformations
    X_train = preprocessor.fit_transform(X_train)
    X_test = preprocessor.transform(X_test)

    # Train the model
    model = train_model(X_train, y_train)
    evaluate_model(model, X_test, y_test)
    save_model_and_preprocessor(model, preprocessor)  # Save both model and preprocessor

    # Run FastAPI
    if "ipykernel" in sys.modules:  
        from IPython.display import display, Javascript
        display(Javascript('window.open("http://127.0.0.1:8000")'))
        uvicorn.run("ML:app", host="0.0.0.0", port=8000, reload=True)
    else:
        uvicorn.run(app, host="0.0.0.0", port=8000)

if __name__ == "__main__":
    main()