# ⚽ Football Match Outcome Predictor

## 📌 Overview
This project is a **Machine Learning-based Football Match Outcome Predictor** built using **FastAPI**, **Scikit-Learn**, and **Random Forest Classifier**. It predicts match results based on statistical data, providing insights for football enthusiasts and analysts.

<img src="https://i.postimg.cc/37PnQw8n/Image-from.png" alt="Ui Implementation">


The project includes:
- **Data Preprocessing & Feature Engineering**
- **Model Training with GridSearchCV**
- **Evaluation Metrics & Visualization**
- **FastAPI Integration for Real-time Predictions**
- **Deployment on Render**

## 🚀 Tech Stack
- **Programming Language:** Python
- **Web Framework:** FastAPI
- **Machine Learning:** Scikit-Learn
- **Data Handling:** Pandas, NumPy
- **Visualization:** Matplotlib, Seaborn
- **Deployment:** Render

## 🌟 Features
- **Predict match outcomes** based on key match statistics
- **Comprehensive data preprocessing** to handle missing values and categorical encoding
- **Model evaluation** with classification metrics, confusion matrices, and ROC curves
- **FastAPI endpoints** to make predictions via API requests
- **CORS support** for frontend integration

## 🏗 Project Structure
```
📂 Football-Match-Predictor              
 ├── ML.py                    # Main ML pipeline & FastAPI app
 ├── requirements.txt         # Dependencies
 ├── README.md                # Project documentation
 ├── model.pkl                # Saved trained model
 ├── preprocessor.pkl         # Saved preprocessor
```

## 🔧 Installation
### 1️⃣ Clone the Repository
```bash
git clone https://github.com/bekizod/test.git
cd test
```

### 2️⃣ Create a Virtual Environment & Install Dependencies
```bash
python -m venv venv
source venv/bin/activate   # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 3️⃣ Run the Application
```bash
python ML.py
```

The FastAPI server will be available at: **http://127.0.0.1:8000**

## 🖥 API Endpoints
### 🚀 Predict Match Outcome
**Endpoint:** `/predict/`
```json
POST http://127.0.0.1:8000/predict/
{
  "home_team_possession": 55,
  "home_team_shots_on_target": 6,
  "home_team_goal_count": 2,
  "away_team_possession": 45,
  "away_team_shots_on_target": 3,
  "away_team_goal_count": 1
}
```
### 🏆 Predict Team vs Team Match Outcome
**Endpoint:** `/predict_match/`
```json
POST http://127.0.0.1:8000/predict_match/
{
  "home_team": {
    "home_team_possession": 55,
    "home_team_shots_on_target": 6,
    "home_team_yellow_cards": 2,
    "home_team_red_cards": 0,
    "home_team_goal_count": 1,
    "home_team_fouls": 10
  },
  "away_team": {
    "home_team_possession": 45,
    "home_team_shots_on_target": 4,
    "away_team_yellow_cards": 3,
    "away_team_red_cards": 1,
    "away_team_goal_count": 2,
    "away_team_fouls": 12
  }
}
```

## 📊 Model Performance
- **Accuracy:** 85%
- **Precision:** 83%
- **Recall:** 80%
- **F1 Score:** 81%

## 🚀 Deployment
This project is deployed on **Render** for easy API access.
🔗 **Live API URL:** [https://test-sjrn.onrender.com](https://test-sjrn.onrender.com)

## 🎯 Frontend UI
Built using Next.js, PrimeReact & TailwindCSS
🔗 **Live Demo:** [Soccer Predictor](https://soccer-predictor-ml-bekizod.netlify.app/)


## 🛠 Future Improvements
✅ Add more advanced ML models like XGBoost or Neural Networks  
✅ Improve feature selection using SHAP values  
✅ Enhance API security & authentication  
✅ Integrate with a frontend dashboard  

## 👨‍💻 Author
Developed by **Bekizod**  
📩 Reach me at: [bekizodcancer@gmail.com](mailto:bekizodcancer@gmail.com)  
🔗 GitHub: [https://github.com/bekizod](https://github.com/bekizod)  
🔗 LinkedIn: [https://www.linkedin.com/in/bereket-wamanuel-73b9712a5](https://www.linkedin.com/in/bereket-wamanuel-73b9712a5)  

## ⭐ Contributing
Feel free to contribute! Fork the repo, create a branch, make changes, and submit a PR.
🚀 If you like this project, don't forget to give it a ⭐ on GitHub! 😊
