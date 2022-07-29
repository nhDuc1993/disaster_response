# Disaster Response Webapp

The project is a Machine Learning Pipeline exercise for analyzing messages in real life disaster situations. The purpose is to help people or organization to categorize messages in life threatening circumstances.

## Dependencies
- Python
- Machine Learning Libraries: NumPy, Pandas, Sciki-Learn
- Natural Language Process Libraries: NLTK
- SQLlite Database Libraqries: SQLalchemy
- Model Loading and Saving Library: Pickle
- Web App and Data Visualization: Flask, Plotly

## Files in the repository

- app
  - template
    - master.html # main page of web app
    - go.html # classification result page of web app
  - run.py # Flask file that runs app
- data
  - disaster_categories.csv # data to process
  - disaster_messages.csv # data to process
  - etl_pipeline.ipynb
  - etl_pipeline.py
  - DisasterResponse.db # database to save clean data to
- models
  - train.py
- README.md

## Github repo
https://github.com/nhDuc1993/disaster_response/

## Tutorial
Clone the repo and
- Cd to data folder and run the ELT pipeline: python etl_pipeline.py
- Cd to model folder and run the ML pipeline: python train.py
- Cd to app folder and run the webapp: python run.py
- Access the webapp on localhost:3000

## Acknowledgements
- Instruction: https://www.udacity.com/
- Data Source: https://www.figure-eight.com/
