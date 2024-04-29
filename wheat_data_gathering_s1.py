import pandas as pd
import os

#Wheat Data

current_dir = os.path.dirname(__file__)

# Construct the path to the Excel files
futmod_wheat_path = os.path.join(current_dir, 'futmodwheat_structured.xlsx')
wheat_data_recent_path = os.path.join(current_dir, 'wheat_data_recent_structured.xlsx')

# Function to load all sheets from an Excel file into a dictionary of DataFrames
def load_excel_sheets(file_path):
    return pd.read_excel(file_path, sheet_name=None)

# Load all sheets for each Excel file
futmod_wheat_dfs = load_excel_sheets(futmod_wheat_path)
wheat_data_recent_dfs = load_excel_sheets(wheat_data_recent_path)

# Display the first few rows of each DataFrame to confirm proper loading and initial view of the data
print("Futures Model Wheat Data:")
for sheet_name, df in futmod_wheat_dfs.items():
    print(f"Sheet: {sheet_name}")
    print(df.head(), "\n")

print("Wheat Data Recent:")
for sheet_name, df in wheat_data_recent_dfs.items():
    print(f"Sheet: {sheet_name}")
    print(df.head(), "\n")


#Saving DFs to Files

output_directory = current_dir

# Function to save a single DataFrame to a CSV file, with an option to include the index
def save_dataframe_to_csv(df, file_name, directory_path, include_index=False):
    file_path = os.path.join(directory_path, file_name)
    df.to_csv(file_path, index=include_index)  # Use the include_index flag to determine whether to include the index

# Function to save each DataFrame from the dictionaries of loaded Excel sheets to CSV files
def save_all_dataframes_to_csv(dataframe_dict, directory_path, prefix='', include_index=False):
    for sheet_name, df in dataframe_dict.items():
        sanitized_sheet_name = sheet_name.replace('/', '_').replace(' ', '_').replace('.', '')
        file_name = f"{prefix}{sanitized_sheet_name}.csv"
        save_dataframe_to_csv(df, file_name, directory_path, include_index)

# Save the wheat data, not including their indexes
save_all_dataframes_to_csv(futmod_wheat_dfs, output_directory, prefix='futmod_')
save_all_dataframes_to_csv(wheat_data_recent_dfs, output_directory, prefix='recent_')