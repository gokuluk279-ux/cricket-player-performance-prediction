# Cricket Player Performance Prediction

A machine-learning sports analytics project that investigates player performance in
One Day International (ODI) cricket under different match conditions and phases.

This repository is based on my MSc Data Science dissertation at the University of
Surrey: **Advanced Predictive Analytics in Cricket – Using Machine Learning to
Optimise Player Selection**.

## Project objective

The project analyses historical ball-by-ball ODI data to:

- identify player-performance indicators;
- compare performance across Powerplay, Middle Overs and Death Overs;
- engineer batting and match-context features;
- compare regression models for player-performance prediction;
- demonstrate how predictive analytics can support selection and tactical decisions.

## Methods and technologies

- Python
- pandas and NumPy
- scikit-learn
- XGBoost
- matplotlib and seaborn
- Jupyter Notebook
- CRISP-DM methodology

Models explored include:

- Linear Regression
- Decision Tree Regressor
- Random Forest Regressor
- XGBoost Regressor

The models were evaluated using MAE, MSE, RMSE and R².

## Repository structure

```text
cricket-player-performance-prediction/
├── data/
│   ├── odi_match_data_sample.csv
│   └── README.md
├── notebooks/
│   └── cricket_player_performance_analysis.ipynb
├── src/
│   ├── data_preparation.py
│   └── model_training.py
├── images/
├── .gitignore
├── LICENSE
├── README.md
└── requirements.txt
```

## Dataset

The analysis uses ball-by-ball ODI match data with fields including match, venue,
innings, ball, batting team, bowling team, striker, bowler, runs, extras and wickets.

The complete dataset contains more than one million records and is not included because
of repository-size limits. A 20,000-row sample is included so that users can inspect the
schema and test the workflow.

Place the full file at:

```text
data/ODI_Match_Data.csv
```

## Running the project

1. Clone the repository.
2. Create a virtual environment.
3. Install the dependencies:

```bash
pip install -r requirements.txt
```

4. Open the notebook:

```bash
jupyter notebook notebooks/cricket_player_performance_analysis.ipynb
```

The original notebook expects a local ODI data file. Update the data path where needed,
or use the included sample file for exploration.

## Key analytical ideas

### Match phases

- Powerplay: overs 0–10
- Middle overs: overs 11–40
- Death overs: overs 41–50

### Player indicators

- total runs;
- balls faced;
- batting average;
- strike rate;
- dismissals;
- performance by match phase;
- venue and opposition context.

## Current limitations

This is a portfolio version of an academic project. The current notebook should be
interpreted as exploratory research rather than a production system. Future improvements
include:

- time-based or match-grouped validation to prevent information leakage;
- reusable preprocessing and modelling pipelines;
- hyperparameter optimisation;
- explainability with SHAP;
- automated tests;
- deployment as an interactive application;
- incorporation of live, tracking, pitch and weather data.

## Author

**Gokul Arumugam**  
MSc Data Science, University of Surrey  
Sports Data Analyst  
Liverpool, United Kingdom

## Licence

This repository's original code and documentation are released under the MIT Licence.
The cricket dataset may be subject to its original provider's terms and is not relicensed
by this repository.
