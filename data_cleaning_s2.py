#Parsing and Cleaning Data

import pandas as pd
import os

# Function to load all CSV files from a directory into a dictionary of DataFrames
def load_all_csv_files(directory_path):
    dataframes = {}
    # List all files in the directory
    for file in os.listdir(directory_path):
        if file.endswith('.csv'):
            file_path = os.path.join(directory_path, file)
            # Use the filename (without extension) as the key in the dictionary
            dataframe_key = os.path.splitext(file)[0]
            dataframes[dataframe_key] = pd.read_csv(file_path)
    return dataframes

# Get the current directory where the script is located
current_dir = os.path.dirname(__file__)

# Load all CSV files in the directory
all_dataframes = load_all_csv_files(current_dir)

# Print the keys of the dictionary to see what DataFrames have been loaded
print("Loaded DataFrames:", list(all_dataframes.keys()))

#Print the first few rows of each DataFrame to confirm they're loaded correctly
for name, df in all_dataframes.items():
    print(f"\n{name} DataFrame:")
    print(df.head())

#Cleaning Futmod DFs 

#Dfs except WASDE_Projection

#Year Column -> Datetime and Single Year Format

def convert_to_year_only(df):
    year_col = df.columns[0]  # Assuming the first column is the year column
    # First, ensure all data in the year column is treated as string for processing
    df[year_col] = df[year_col].astype(str)
    
    try:
        # Check if there's a range indicated by '-' and split on it, taking the first part
        if any('-' in x for x in df[year_col]):  # This checks each element safely
            df[year_col] = df[year_col].apply(lambda x: x.split('-')[0])
        
        # Convert to datetime and extract the year
        df[year_col] = pd.to_datetime(df[year_col], errors='coerce').dt.year
    except Exception as e:
        print(f"Error processing the year column in {year_col}: {str(e)}")
        # If conversion fails, attempt to force numeric interpretation
        df[year_col] = pd.to_numeric(df[year_col], errors='coerce')

    df.rename(columns={year_col: 'year'}, inplace=True)
    return df

futmod_dfs_to_clean = {key: df for key, df in all_dataframes.items() if 'futmod' in key.lower() and 'wasde_projections' not in key.lower()}

for key, df in futmod_dfs_to_clean.items():
    print(f"Cleaning {key}")
    df_cleaned = convert_to_year_only(df)
    print(df_cleaned.head())

# Save the cleaned DataFrames back to CSV
output_directory = os.path.dirname(__file__)  
for key, df in futmod_dfs_to_clean.items():
    output_path = os.path.join(output_directory, f"{key}_cleaned.csv")
    df.to_csv(output_path, index=False)
    print(f"Saved cleaned data for {key} to {output_path}")

# Specific cleaning for futmod_WASDE_Projections DataFrame

def clean_wasde_projections(df):
    # Print columns before cleaning for debugging
    print("Columns before cleaning:", df.columns.tolist())
    
    # Normalize column names by stripping any leading/trailing spaces
    df.columns = [col.strip() for col in df.columns]
    
    # Remove unwanted columns including 'year'
    columns_to_remove = ['Unnamed: 3', 'Unnamed: 5', 'Unnamed: 6', 'WASDE release date', 'year']
    df.drop(columns=[col for col in columns_to_remove if col in df.columns], axis=1, inplace=True)
    
    # Convert 'period' to datetime
    if 'period' in df.columns:
        df['period'] = pd.to_datetime(df['period'], errors='coerce')
    
    # Print columns after cleaning for debugging
    print("Columns after cleaning:", df.columns.tolist())
    return df

# Check and clean the DataFrame
if 'futmod_WASDE_Projections' in all_dataframes:
    print("\nCleaning futmod_WASDE_Projections DataFrame:")
    wasde_df = clean_wasde_projections(all_dataframes['futmod_WASDE_Projections'])
    print(wasde_df.head())

    # Save the cleaned DataFrame back to CSV
    output_path = os.path.join(current_dir, 'futmod_WASDE_Projections_cleaned.csv')
    wasde_df.to_csv(output_path, index=False)
    print(f"Saved cleaned WASDE Projections data to {output_path}")

#Cleaning Recent Wheat Data DFs

def convert_marketing_year(df):
    # Assuming 'Marketing Year' is the column of interest
    if 'Marketing Year' in df.columns:
        # Normalize the column for safety
        df['Marketing Year'] = df['Marketing Year'].astype(str)
        if '/' in df['Marketing Year'].iloc[0]:
            # Handle as '2019/2020', extract the first year
            df['Marketing Year'] = pd.to_datetime(df['Marketing Year'].apply(lambda x: x.split('/')[0]), format='%Y').dt.year
        else:
            # Direct conversion to datetime
            df['Marketing Year'] = pd.to_datetime(df['Marketing Year'], format='%Y', errors='coerce').dt.year

    return df

for key, df in all_dataframes.items():
    if 'recent' in key.lower() and 'world' not in key.lower():
        print(f"Cleaning {key}")
        df_cleaned = convert_marketing_year(df)
        print(df_cleaned[['Marketing Year']].head())

output_directory = os.path.dirname(__file__)
for key, df in all_dataframes.items():
    if 'recent' in key.lower() and 'world' not in key.lower():
        output_path = os.path.join(output_directory, f"{key}_cleaned.csv")
        df.to_csv(output_path, index=False)
        print(f"Saved cleaned data for {key} to {output_path}")


#Clean Climate Data

# Function to clean temperature and precipitation DataFrames
def clean_temp_precip_data(df):
    if 'date' in df.columns and not pd.api.types.is_datetime64_any_dtype(df['date']):
        df['date'] = pd.to_datetime(df['date'], errors='coerce')

    # Drop 'attributes' column if it exists
    if 'attributes' in df.columns:
        df.drop('attributes', axis=1, inplace=True)
    
    return df

if 'tavg_kansas_cimarron' in all_dataframes:
    temperature_2020_df = clean_temp_precip_data(all_dataframes['tavg_kansas_cimarron'])
    print("Cleaned 'tavg_kansas_cimarron' DataFrame:")
    print(temperature_2020_df.head())

    output_path = os.path.join(current_dir, 'tavg_kansas_cimarron_cleaned.csv')
    temperature_2020_df.to_csv(output_path, index=False)
    print(f"Saved cleaned 'tavg_kansas_cimarron' data to {output_path}")

if 'precipitation_kansas' in all_dataframes:
    precipitation_2020_df = clean_temp_precip_data(all_dataframes['precipitation_kansas'])
    print("Cleaned 'precipitation_kansas' DataFrame:")
    print(precipitation_2020_df.head())

    output_path = os.path.join(current_dir, 'precipitation_kansas_cleaned.csv')
    precipitation_2020_df.to_csv(output_path, index=False)
    print(f"Saved cleaned 'precipitation_kansas' data to {output_path}")
