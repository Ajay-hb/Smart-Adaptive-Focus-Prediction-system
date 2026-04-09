# Smart Adaptive Focus Prediction System

A machine learning project that predicts a user's focus score based on lifestyle and environment factors such as sleep, screen time, stress, mood, and task difficulty.

This repository includes:
- a Jupyter notebook for experimentation and model workflow,
- and a Streamlit web app for interactive predictions.

## Demo Purpose

The app is designed to:
- estimate a focus score (`0-100`) from user inputs,
- provide quick improvement suggestions,
- and demonstrate an end-to-end ML workflow from data generation to deployment UI.

## Features

- Synthetic dataset generation (`500` rows by default)
- Feature engineering:
  - `sleep_category`
  - `sleep_stress_interaction`
  - `caffeine_sleep_interaction`
- Label encoding for categorical features
- XGBoost regressor for focus score prediction
- Streamlit-based interactive interface
- Personalized "what to improve" suggestions

## Tech Stack

- Python
- Pandas
- NumPy
- scikit-learn
- XGBoost
- Streamlit

## Project Structure

```text
Smart Adaptive Focus Prediction System/
├── app.py
├── requirements.txt
├── Smart_Adaptive_Focus_Prediction_System.ipynb
└── README.md
```

## Installation

1. Clone the repository:

```bash
git clone https://github.com/<your-username>/<your-repo-name>.git
cd "<your-repo-name>"
```

2. (Recommended) Create and activate a virtual environment:

```bash
python -m venv .venv
```

On Windows (PowerShell):

```powershell
.venv\Scripts\Activate.ps1
```

3. Install dependencies:

```bash
pip install -r requirements.txt
```

## Run the Streamlit App

```bash
streamlit run app.py
```

If `streamlit` command is not recognized:

```bash
python -m streamlit run app.py
```

Open in browser:
- `http://localhost:8501`

## How It Works

1. Generates synthetic focus-related data.
2. Trains an XGBoost regression model.
3. Accepts user inputs from the Streamlit UI.
4. Predicts a focus score.
5. Tests alternative input adjustments and suggests changes that can improve the predicted score.

## Input Parameters

- `sleep_hours` (3-9)
- `screen_time` (1-9)
- `noise_level` (1-3)
- `time_of_day` (1-3)
- `caffeine` (0-3)
- `stress_level` (1-5)
- `exercise` (0-3)
- `mood` (1-5)
- `task_difficulty` (1-3)

## Notes

- The dataset is synthetic and intended for educational/demo use.
- Model predictions are approximate and should not be used as medical or clinical advice.

## Future Improvements

- Save/load trained model artifacts (`joblib`)
- Add model evaluation dashboard (RMSE/MAE/R2 and plots)
- Add user history tracking and trend charts
- Deploy to Streamlit Community Cloud

## License

You can add your preferred license here (for example, MIT).

---

If you found this project useful, consider starring the repository.
