This project was part of an open-ended data analysis project from my DS 710 Programming for Data Science class (UWLAX MS in Data Science)

I chose to analyze the U.S. wheat market dynamics and how certain variables affect domestic price and production.

The Data Provenance PDF file contains information on how I acquired my data from the NOAA and USDA ERS.

1. Data Acquisition (step 1):
  - wheat_data_gathering_s1.py takes the structured excel files I copy/pasted from the raw excel files I downloaded from the USDA ERS
  - climate_data_webscraping_s1.py web scrapes TAVG and Precipitation data from the NOAA
  - Step 1 takes the raw data and converts into CSV files

2. Data Cleaning (step 2):
  - data_cleaning_s2.py takes the CSV files, does some initial cleaning, and exports the dataframes back to CSV files denoted with suffix _cleaned.

3. Data Analysis (step 3):
   - wheat_data_analysis_s3.py takes the cleaned files and processes them through different Python code to generate different plots and numerical data
   - The generated .png and .txt files contain the plots and numerical data corresponding to 4 of the exploratory questions I had
  
  The Executive Report contains my project summary, experience, and analysis of the 4 questions based on the generated visuals and numerical data. 
