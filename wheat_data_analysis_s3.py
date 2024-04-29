import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import statsmodels.api as sm

current_dir = os.path.dirname(__file__)

first_name = "Tej" 
last_name  = "Rai" 

#Analysis

#1
#Compare WASDE projections to the actual U.S. farmer price recieved for wheat 
#Futmod Wasde & Futmod Historical Farm Prices (June - December 2023)

#Load Data
wasde_projections_path = 'futmod_WASDE_Projections_cleaned.csv'
us_farm_prices_path = 'futmod_Farm_Prices_cleaned.csv'

wasde_df = pd.read_csv(wasde_projections_path)
farm_prices_df = pd.read_csv(us_farm_prices_path)

# Convert 'period' in wasde_df to datetime and filter for 2023
wasde_df['period'] = pd.to_datetime(wasde_df['period'])
wasde_2023_df = wasde_df[wasde_df['period'].dt.year == 2023].copy()

# Group WASDE projections by month and calculate the average projection for each month
wasde_2023_df['month'] = wasde_2023_df['period'].dt.month
monthly_wasde_avg = wasde_2023_df.groupby('month')['WASDE projection'].mean().reset_index()

# Filter the farm prices for the year 2023 and select June to December
farm_prices_2023 = farm_prices_df[farm_prices_df['year'] == 2023].copy()
months_of_interest = ['June', 'July', 'August', 'September', 'October', 'November', 'December']
farm_prices_2023 = farm_prices_2023.loc[:, months_of_interest]

# To align with the monthly average WASDE data, create a new DataFrame with these months and corresponding prices
monthly_farm_prices = pd.DataFrame({
    'month': range(6, 13),  # Months June to December
    'Farm price': farm_prices_2023.iloc[0].values
})

# Merge the average WASDE data with the actual farm prices on the month
wasde_vs_actual_df = pd.merge(monthly_wasde_avg, monthly_farm_prices, on='month')

# Convert the 'month' column to string for better plotting
wasde_vs_actual_df['month'] = wasde_vs_actual_df['month'].astype(str)

wasde_vs_actual_df.to_csv('Q1_wasde_vs_actual.csv', index=False)

fig, ax = plt.subplots()
sns.lineplot(data=wasde_vs_actual_df, x='month', y='WASDE projection', marker='o', label='WASDE Projection', color='blue')
sns.lineplot(data=wasde_vs_actual_df, x='month', y='Farm price', marker='x', label='Farm Price', color='red')

plt.savefig('Q1_wasde_vs_actual_price.png')

# ----------------------------------------

#2 

#How do weather patterns impact the price of wheat?
#Determine a correlation between price with avg. precipitation and avg. temperature (Jan - Dec 2017 - 2020)

#2.1

#Prepare farm_prices_df 

# Load farm prices data
original_df = pd.read_csv('futmod_Farm_Prices.csv')

# Drop the "Annual" column
original_df = original_df.drop(columns=['Annual'])

# Remove rows from 1975-76 through 2015-2016
# Assuming the 'year' column is in format 'YYYY-YY', extract the start year and filter out those not needed
original_df['start_year'] = original_df['year'].apply(lambda x: int(x[:4]))
cleaned_df = original_df[original_df['start_year'] > 2015].drop(columns=['start_year'])

# Adjusting the data so that each year aligns properly with months from January to December
adjusted_years_df = pd.DataFrame()

for i in range(len(cleaned_df) - 1):
    year_data = {}
    year_data['year'] = cleaned_df.iloc[i + 1]['year'].split('-')[0]  # Get the next year as the year label
    year_data.update(cleaned_df.iloc[i][['January', 'February', 'March', 'April', 'May']].to_dict())
    year_data.update(cleaned_df.iloc[i + 1][['June', 'July', 'August', 'September', 'October', 'November', 'December']].to_dict())
    adjusted_years_df = pd.concat([adjusted_years_df, pd.DataFrame([year_data])], ignore_index=True)

adjusted_years_df = adjusted_years_df[adjusted_years_df['year'].astype(int) <= 2020]  # Filter up to the year 2020

# Use the adjusted farm prices data
farm_prices_df = adjusted_years_df

# Reshape farm prices data
months = ['January', 'February', 'March', 'April', 'May', 'June', 'July', 'August', 'September', 'October', 'November', 'December']
monthly_farm_prices = farm_prices_df.melt(id_vars='year', value_vars=months, var_name='month', value_name='price')
monthly_farm_prices['month'] = pd.to_datetime(monthly_farm_prices['month'], format='%B').dt.month

# Convert the 'year' from string to integer for consistency in merging
monthly_farm_prices['year'] = monthly_farm_prices['year'].astype(int)

# Load temperature and precipitation data
temperature_kansas_path = 'tavg_kansas_cimarron_cleaned.csv'
precipitation_kansas_path = 'precipitation_kansas_cleaned.csv'

temp_df = pd.read_csv(temperature_kansas_path)
prcp_df = pd.read_csv(precipitation_kansas_path)

# Convert 'date' columns to datetime
temp_df['date'] = pd.to_datetime(temp_df['date'])
prcp_df['date'] = pd.to_datetime(prcp_df['date'])

# Group by year and month, and calculate the average
temp_df['month'] = temp_df['date'].dt.month
temp_df['year'] = temp_df['date'].dt.year
monthly_temp_avg = temp_df.groupby(['year', 'month'])['value'].mean().reset_index()
monthly_temp_avg.rename(columns={'value': 'avg_temp'}, inplace=True)

prcp_df['month'] = prcp_df['date'].dt.month
prcp_df['year'] = prcp_df['date'].dt.year
monthly_prcp_avg = prcp_df.groupby(['year', 'month'])['value'].mean().reset_index()
monthly_prcp_avg.rename(columns={'value': 'avg_prcp'}, inplace=True)

# Merge the datasets
climate_prices_df = pd.merge(monthly_temp_avg, monthly_prcp_avg, on=['year', 'month'], how='inner')
climate_prices_df = pd.merge(climate_prices_df, monthly_farm_prices, on=['year', 'month'], how='inner')

# Apply additional filters to exclude specific months for certain years
# Exclude months 1-5 for year 2017 and months 6-12 for year 2020
filtered_climate_prices_df = climate_prices_df[
    ~((climate_prices_df['year'] == 2017) & (climate_prices_df['month'].isin([1, 2, 3, 4, 5]))) &
    ~((climate_prices_df['year'] == 2020) & (climate_prices_df['month'].isin([6, 7, 8, 9, 10, 11, 12])))
]

# Filter for years 2017 to 2020
filtered_climate_prices_df = filtered_climate_prices_df[
    filtered_climate_prices_df['year'].isin([2017, 2018, 2019, 2020])
]

# Reset index
filtered_climate_prices_df.reset_index(drop=True, inplace=True)

filtered_climate_prices_df.to_csv('Q2_price_by_prcp_&_tavg.csv', index=False)

# Calculate the correlation matrix
correlation_matrix = filtered_climate_prices_df.corr()

# Use Seaborn to plot the correlation matrix with numerical values
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, fmt=".2f", cmap='coolwarm', square=True, cbar_kws={"shrink": .5})
plt.xticks(rotation=45, ha='right')
plt.yticks(rotation=0)

plt.savefig('Q2-1_heatplot_price_by_prcp_tavg.png')

from scipy.stats import pearsonr

# Perform Pearson's r test for correlation significance
avg_temp = filtered_climate_prices_df['avg_temp'].dropna()
price = filtered_climate_prices_df.loc[avg_temp.index, 'price']  # Make sure to align the indices
corr_coef, p_value = pearsonr(avg_temp, price)

print("\n================ Q2.1 Numerical Results ================\n")


print(f"Correlation coefficient between average temperature and price: {corr_coef:.2f}")
print(f"P-value of the correlation: {p_value:.3f}")

#2.2

#Time Series Analysis to identify trends and seasonality in wheat prices and weather pattens

from statsmodels.tsa.seasonal import seasonal_decompose

# Make sure 'year' and 'month' are used to create a datetime index
filtered_climate_prices_df['date'] = pd.to_datetime(filtered_climate_prices_df['year'].astype(str) + '-' + filtered_climate_prices_df['month'].astype(str))
filtered_climate_prices_df.set_index('date', inplace=True)

# Ensure the data is sorted by date
filtered_climate_prices_df.sort_index(inplace=True)

# Select the farm price for decomposition
ts_data = filtered_climate_prices_df['price']

# Decompose the time series
result = seasonal_decompose(ts_data, model='additive', period=12)  # assuming monthly data, hence period=12

# Plot the decomposed components
fig, axes = plt.subplots(4, 1, figsize=(10, 8))
axes[0].plot(result.observed)
axes[0].set_title('Observed')
axes[1].plot(result.trend)
axes[1].set_title('Trend')
axes[2].plot(result.seasonal)
axes[2].set_title('Seasonal')
axes[3].plot(result.resid)
axes[3].set_title('Residual')
plt.tight_layout()

plt.savefig('Q2-2_timeplot_price_by_prcp_tavg.png')

from statsmodels.stats.diagnostic import acorr_ljungbox

# Numerical Analysis

print("\n================ Q2.2 Time Series Numerical Analysis ================\n")


trend_mean = result.trend.mean()
trend_std = result.trend.std()
seasonal_amplitude = (result.seasonal.max() - result.seasonal.min()) / 2
residual_mean = result.resid.mean()
residual_std = result.resid.std()

print('Trend Mean:', trend_mean)
print('Trend Standard Deviation:', trend_std)
print('Seasonal Amplitude:', seasonal_amplitude)
print('Residual Mean:', residual_mean)
print('Residual Standard Deviation:', residual_std)

# Statistical testing on residuals
non_na_residuals = result.resid.dropna()
nobs = len(non_na_residuals)
max_lags = min(12, nobs - 1)
lb_results = acorr_ljungbox(non_na_residuals, lags=[max_lags], return_df=True)
print('Ljung-Box test statistic:', lb_results['lb_stat'].iloc[0])
print('Ljung-Box test p-value:', lb_results['lb_pvalue'].iloc[0])

# ----------------------------------------

#3

#How U.S. Wheat exports and foreign wheat prices effect domestic wheat prices received by farmers? (2020 - 2023)

# Load the data
exports_path = 'recent_Wheat_Exports_cleaned.csv'
foreign_domestic_prices_path = 'recent_US_&_Foreign_Wheat_Prices_cleaned.csv'
us_prices_path = 'recent_Wheat_Avg_Price_by_Farmer_cleaned.csv'

exports_df = pd.read_csv(exports_path)
foreign_domestic_prices_df = pd.read_csv(foreign_domestic_prices_path)
us_prices_df = pd.read_csv(us_prices_path)


# Take data from 2020-2023 for exports_df
exports_df = exports_df[exports_df['Marketing Year'].isin([2020, 2021, 2022, 2023])]

exports_df.rename(columns={'U.S. exports million bushels': 'Exports_MillionBushels'}, inplace=True)

combined_df = pd.merge(exports_df, foreign_domestic_prices_df, how='inner', on='Marketing Year')
combined_df = pd.merge(combined_df, us_prices_df, how='inner', on='Marketing Year')

# Rename columns
combined_df.rename(columns={'Avg 2/': 'Foreign Price', 'Wt avg 2/': 'Domestic Price'}, inplace=True)

# Ensure 'Marketing Year' is datetime-like
combined_df['Marketing Year'] = pd.to_datetime(combined_df['Marketing Year'], format='%Y')

# Drop rows with missing values in the 'Exports_MillionBushels' column
combined_df.dropna(subset=['Exports_MillionBushels'], inplace=True)

# Convert foreign prices from dollars per metric ton to dollars per bushel
# Conversion factor: 1 metric ton = 36.74 bushels
combined_df['Foreign Price'] /= 36.74

# Select only the required columns
final_columns = ['Marketing Year', 'Exports_MillionBushels', 'Foreign Price', 'Domestic Price']
combined_df = combined_df[final_columns]

combined_df.to_csv('Q3_domesticprice_by_exports_&_foreignprice.csv', index=False)

# Simple Linear Regression for 'Exports_MillionBushels'
X_exports = sm.add_constant(combined_df['Exports_MillionBushels'])  # adding a constant
y = combined_df['Domestic Price']
model_exports = sm.OLS(y, X_exports).fit()

# Simple Linear Regression for 'Foreign Price'
X_foreign = sm.add_constant(combined_df['Foreign Price'])  # adding a constant
model_foreign = sm.OLS(y, X_foreign).fit()

# Plotting
fig, ax = plt.subplots(1, 2, figsize=(14, 5))

# Plot the linear regression plots
fig, ax = plt.subplots(1, 2, figsize=(14, 5))
ax[0].scatter(combined_df['Exports_MillionBushels'], y, color='blue')
ax[0].plot(combined_df['Exports_MillionBushels'], model_exports.fittedvalues, color='red')
ax[0].set_title('Domestic Price vs Exports')
ax[0].set_xlabel('Exports (Million Bushels)')
ax[0].set_ylabel('Domestic Price (USD)')
ax[1].scatter(combined_df['Foreign Price'], y, color='green')
ax[1].plot(combined_df['Foreign Price'], model_foreign.fittedvalues, color='red')
ax[1].set_title('Domestic Price vs Foreign Price')
ax[1].set_xlabel('Foreign Price (USD)')
ax[1].set_ylabel('Domestic Price (USD)')
plt.tight_layout()

plt.savefig('Q3_linearplots_domesticprice_by_exports_foreignprice.png')


print("\n================ Q3 Simple Linear Model ================\n")

# Output the summaries of the models
print("Exports Model Summary:")
print(model_exports.summary())
print("\nForeign Price Model Summary:")
print(model_foreign.summary())

# ----------------------------------------

#4 

 #How is U.S. wheat production impacted by U.S. Wheat food use and wheat exportation?

# Load the datasets
food_use_path = 'recent_Wheat_Food_Use_cleaned.csv'
prod_exports_path = 'recent_Wheat_Exports_cleaned.csv'

# Clean and prepare the food use data
food_use_df = pd.read_csv(food_use_path)
food_use_cleaned = food_use_df[['Marketing Year', 'Annual']]
food_use_cleaned['Marketing Year'] = pd.to_datetime(food_use_cleaned['Marketing Year'], format='%Y')

# Clean and prepare the production and exports data
prod_exports_df = pd.read_csv(prod_exports_path)
prod_exports_df = prod_exports_df[['Marketing Year', 'U.S. production million bushels', 'U.S. exports million bushels']].copy()
prod_exports_df['U.S. production million bushels'] *= 1000  # Convert to 1000 bushels to match food use units
prod_exports_df['U.S. exports million bushels'] *= 1000  # Convert to 1000 bushels
prod_exports_df['Marketing Year'] = pd.to_datetime(prod_exports_df['Marketing Year'], format='%Y')

# Merge the datasets on 'Marketing Year'
combined_df = pd.merge(food_use_cleaned, prod_exports_df, on='Marketing Year', how='inner')

# Exclude the year 2023 from the analysis
combined_df = combined_df[combined_df['Marketing Year'].dt.year != 2023]

# Rename columns for clarity
combined_df.rename(columns={'U.S. production million bushels': 'U.S. Production (1000 Bushels)', 'U.S. exports million bushels': 'U.S. Exports (1000 Bushels)', 'Annual': 'Annual Food Use (1000 Bushels)'}, inplace=True)

# Save the merged data to a new CSV file
combined_df.to_csv('Q4_prod_by_food_use_and_export.csv', index=False)

# Plot the time series data
fig, ax1 = plt.subplots(figsize=(15, 7))
ax1.plot(combined_df['Marketing Year'].dt.year, combined_df['Annual Food Use (1000 Bushels)'], label='Food Use (1000 Bushels)', marker='o', color='tab:blue')
ax1.plot(combined_df['Marketing Year'].dt.year, combined_df['U.S. Exports (1000 Bushels)'], label='Exports (1000 Bushels)', marker='^', color='tab:red')
ax2 = ax1.twinx()
ax2.plot(combined_df['Marketing Year'].dt.year, combined_df['U.S. Production (1000 Bushels)'], label='Production (1000 Bushels)', marker='x', color='tab:green')
ax1.set_xlabel('Year')
ax1.set_ylabel('Food Use and Exports (1000 Bushels)', color='tab:blue')
ax2.set_ylabel('Production (1000 Bushels)', color='tab:green')
ax1.set_title('U.S. Wheat Food Use, Production, and Exports (2004 - 2022)')
ax1.legend(loc='upper left')
ax2.legend(loc='upper right')
ax1.grid(True)
ax1.set_xticks(combined_df['Marketing Year'].dt.year)
ax1.set_xticklabels(combined_df['Marketing Year'].dt.year, rotation=45)

plt.savefig('Q4_lineplot_prod_by_exports_fooduse.png')

# Linear Regression: Predicting Production based on Food Use and Exports
# Adding a constant for the intercept
X = combined_df[['Annual Food Use (1000 Bushels)', 'U.S. Exports (1000 Bushels)']]
y = combined_df['U.S. Production (1000 Bushels)']
X = sm.add_constant(X)

# Fit the model
production_model = sm.OLS(y, X).fit()

# Output the summary of the regression model
print("\n================ Q4 Regression Model Summary ================\n")
print(production_model.summary())

# Plotting for visual inspection of relationships
plt.figure(figsize=(10, 5))
plt.scatter(X['Annual Food Use (1000 Bushels)'], y, color='blue', label='Food Use vs Production')
plt.scatter(X['U.S. Exports (1000 Bushels)'], y, color='red', label='Exports vs Production')
plt.legend()
plt.title('Scatter Plot of Variables')
plt.xlabel('Food Use and Exports (in thousands of bushels)')
plt.ylabel('Production (in thousands of bushels)')

plt.savefig('Q4_scatter_plot_prod_by_exports_fooduse.png')


# Calculate correlation matrix with production as the dependent variable
print("\n================ Q4 Correlation Matrix ================\n")
correlation_matrix = combined_df[['Annual Food Use (1000 Bushels)', 'U.S. Production (1000 Bushels)', 'U.S. Exports (1000 Bushels)']].corr()
print(correlation_matrix)


# ----------------------------------------

#File Writing

# Q2 Numerical Analysis
with open('Q2_numerical_analysis.txt', 'w') as file:
    file.write("================ Q2.1 Numerical Results ================\n\n")
    file.write(f"Correlation coefficient between average temperature and price: {corr_coef:.2f}\n")
    file.write(f"P-value of the correlation: {p_value:.3f}\n\n")
    file.write("\n================ Q2.2 Time Series Numerical Analysis ================\n\n")
    file.write('Trend Mean: {}\n'.format(trend_mean))
    file.write('Trend Standard Deviation: {}\n'.format(trend_std))
    file.write('Seasonal Amplitude: {}\n'.format(seasonal_amplitude))
    file.write('Residual Mean: {}\n'.format(residual_mean))
    file.write('Residual Standard Deviation: {}\n\n'.format(residual_std))
    file.write('Ljung-Box test statistic: {}\n'.format(lb_results['lb_stat'].iloc[0]))
    file.write('Ljung-Box test p-value: {}\n'.format(lb_results['lb_pvalue'].iloc[0]))

# Q3 Regression Models
with open('Q3_regression_models.txt', 'w') as file:
    file.write("Exports Model Summary:\n\n")
    file.write(model_exports.summary().as_text() + "\n\n")
    file.write("Foreign Price Model Summary:\n\n")
    file.write(model_foreign.summary().as_text() + "\n\n")

# Q4 Regression Model
with open('Q4_regression_model.txt', 'w') as file:
    file.write("\n================ Q4 Regression Model Summary ================\n\n")
    file.write(production_model.summary().as_text() + "\n")
