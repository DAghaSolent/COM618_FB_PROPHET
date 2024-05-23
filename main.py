import pandas as pd
from prophet import Prophet
from prophet.plot import plot_plotly
import warnings

warnings.filterwarnings("ignore")

pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)

# Loading the Excel Spreadsheet ready for modelling and forecasting. Table 1 sheet from my Excel dataset.
df = pd.read_excel(r'alcoholspecificdeaths2021.xlsx', sheet_name='Table 1', skiprows=4)

# Loading table 2 from my Excel spreadsheet for modelling and forecasting table 2 data.
df_tbl2 = pd.read_excel(r'alcoholspecificdeaths2021.xlsx', sheet_name='Table 2', skiprows=4)

# Resetting and clearing the data to be processed for the Facebook Prophet Method
df.reset_index(inplace=True)

# Defining a gender list that will be utilised in all the below forecasting prediction FB Prophet methods
genders = ['Females', 'Males', 'Persons']


def model_forecast_deaths_uk_10_years_tbl1():

    for gender in genders:
        # Creating and initialising the FB Prophet Model
        prophet = Prophet(
            daily_seasonality=False,
            yearly_seasonality=True,
            weekly_seasonality=False,
            changepoint_prior_scale=0.05,
            seasonality_prior_scale=10.0
        )

        # Creating a new Data Frame and Filtering the region since I am only interested in UK and the current gender
        df_filtered = df[(df['Area name'] == 'United Kingdom') & (df['Sex'] == gender)]

        # Printing and displaying actual number of deaths before forecasting and predicting future number of deaths
        print(f"Actual number of deaths data for gender: {gender}")
        print(df_filtered[['Year [note 3]', 'Number of deaths']].rename(columns={'Year [note 3]': 'Year'}))

        # Renaming and assigning my columns to specified column names to match with FB Prophet requirements and work
        # with the FB Prophet model successfully.
        df_filtered = df_filtered.rename(columns={'Year [note 3]': 'ds', 'Number of deaths': 'y'})

        # Convert 'ds' which represents the date column to datetime so that it can be read by the fb prophet model.
        df_filtered['ds'] = pd.to_datetime(df_filtered['ds'].astype(str), format='%Y')

        # Fitting the model onto my dataframe containing my dataset.
        prophet.fit(df_filtered)

        # Creating a dataframe to be used for the prediction forecasting
        future = prophet.make_future_dataframe(periods=10, freq='YE')

        # Passing the future Dataframe to generate a forecast prediction on deaths for the next 10 years for each gender
        forecast = prophet.predict(future)

        # Rounding up or down the forecast values to the nearest whole number
        forecast['yhat'] = forecast['yhat'].round()

        # Plot the forecasts that were made by FB Prophet and displaying them to the user using plotly.
        fig = plot_plotly(prophet, forecast)
        fig.update_layout(xaxis_title="Year",
                          yaxis_title="Number of Deaths",
                          title_text=f"FB Prophet Prediction for Alcohol Specific Deaths within the UK - {gender}"
                          )
        fig.show()

        # Outputting and displaying the forecasts to the console.
        print(f"Forecasted Alcohol Specific Deaths within the UK for Gender: {gender}")
        print(forecast[['ds', 'yhat']].rename(columns={'ds': 'Date', 'yhat': 'Number of Deaths'}))

        print("_______________________________________________________________________________________________________")


def model_forecast_deaths_countries_that_make_up_uk_tbl1():
    # Defining the countries that make up UK for forecasting and predicting number of deaths for these countries.
    countries = ['England', 'Scotland', 'Northern Ireland', 'Wales']

    # looping through each country within country_categories
    for country in countries:
        # looping through each gender in gender categories within each country of country categories
        for gender in genders:
            # Creating and initialising the FB Prophet Model
            prophet = Prophet(
                daily_seasonality=False,
                yearly_seasonality=True,
                weekly_seasonality=False,
                changepoint_prior_scale=0.05,
                seasonality_prior_scale=10.0
            )

            # Creating a dataframe and filtering the current country with the current gender.
            df_filtered = df[(df['Area name'] == country) & (df['Sex'] == gender)]

            # Printing and displaying the actual number of deaths for the current country and gender
            print(f"Actual number of deaths for gender: {gender} in country: {country}")
            print(df_filtered[['Year [note 3]', 'Number of deaths']].rename(columns={'Year [note 3]': 'Year'}))

            # Renaming and assigning my columns to specified column names to match with FB Prophet requirements and work
            # with the FB Prophet model successfully.
            df_filtered = df_filtered.rename(columns={'Year [note 3]': 'ds', 'Number of deaths': 'y'})

            # Convert 'ds' which represents the date column to datetime so that it can be read by the fb prophet model.
            df_filtered['ds'] = pd.to_datetime(df_filtered['ds'].astype(str), format='%Y')

            # Fitting the model onto my dataframe containing my dataset.
            prophet.fit(df_filtered)

            # Creating a dataframe to be used for the prediction forecasting
            future = prophet.make_future_dataframe(periods=10, freq='YE')

            # Passing the future Dataframe to generate a forecast prediction on deaths for the next 10 years for each
            # gender within each country that makes up the United Kingdom
            forecast = prophet.predict(future)

            # Rounding up or down the forecast values to the nearest whole number
            forecast['yhat'] = forecast['yhat'].round()

            #Plot the forecasts that were made by FB Prophet and displaying them to the user using plotly.
            fig = plot_plotly(prophet, forecast)
            fig.update_layout(xaxis_title="Year",
                              yaxis_title="Number of Deaths",
                              title_text=f"FB Prophet Prediction for Alcohol Specific Deaths for Gender: {gender} in "
                                         f"Country: {country}"
                              )
            fig.show()

            # Outputting and displaying the forecasts to the console.
            print(f"Forecasted Alcohol Specific Deaths for Gender: {gender} in Country: {country}")
            print(forecast[['ds', 'yhat']].rename(columns={'ds': 'Date', 'yhat': 'Number of Deaths'}))
            print("___________________________________________________________________________________________________")


def model_forecast_deaths_regions_within_england_tbl1():
    # Defining the regions within england for forecasting and predicting number of deaths for these english regions
    eng_regions = ['East', 'East Midlands', 'London', 'North East', 'North West', 'South East', 'South West',
                   'West Midlands', 'Yorkshire and The Humber']

    # looping through each region in England
    for region in eng_regions:
        for gender in genders:
            # Creating and initialising the FB Prophet Model
            prophet = Prophet(
                daily_seasonality=False,
                yearly_seasonality=True,
                weekly_seasonality=False,
                changepoint_prior_scale=0.05,
                seasonality_prior_scale=10.0
            )

            # Creating a dataframe and filtering the current region with the current gender.
            df_filtered = df[(df['Area name'] == region) & (df['Sex'] == gender)]

            # Printing and displaying the actual number of deaths for the current region and gender
            print(f"Actual number of deaths for gender: {gender} in English region: {region}")
            print(df_filtered[['Year [note 3]', 'Number of deaths']].rename(columns={'Year [note 3]': 'Year'}))

            # Renaming and assigning my columns to specified column names to match with FB Prophet requirements and work
            # with the FB Prophet model successfully.
            df_filtered = df_filtered.rename(columns={'Year [note 3]': 'ds', 'Number of deaths': 'y'})

            # Convert 'ds' which represents the date column to datetime so that it can be read by the fb prophet model.
            df_filtered['ds'] = pd.to_datetime(df_filtered['ds'].astype(str), format='%Y')

            # Fitting the model onto my dataframe containing my dataset.
            prophet.fit(df_filtered)

            # Creating a dataframe to be used for the prediction forecasting
            future = prophet.make_future_dataframe(periods=10, freq='YE')

            # Passing the future Dataframe to generate a forecast prediction on deaths for the next 10 years for each
            # gender within each region within England.
            forecast = prophet.predict(future)

            # Rounding up or down the forecast values to the nearest whole number
            forecast['yhat'] = forecast['yhat'].round()

            # Plot the forecasts that were made by FB Prophet and displaying them to the user using plotly.
            fig = plot_plotly(prophet, forecast)
            fig.update_layout(xaxis_title="Year",
                              yaxis_title="Number of Deaths",
                              title_text=f"FB Prophet Prediction for Deaths for Gender: {gender} in "
                                         f"England Region: {region}"
                              )
            fig.show()

            # Outputting and displaying the forecasts to the console.
            print(f"Forecasted Deaths for Gender: {gender} in England Region: {region}")
            print(forecast[['ds', 'yhat']].rename(columns={'ds': 'Date', 'yhat': 'Number of Deaths'}))
            print("___________________________________________________________________________________________________")

def model_forecast_deaths_for_age_groups_tbl2():
    # Defining the age groups for forecasting and predicting the number of deaths for these age groups.
    age_groups = ['01-04', '05-09', '10-14', '15-19', '20-24', '25-29', '30-34', '35-39', '40-44', '45-49', '50-54',
                  '55-59', '60-64', '65-69', '70-74', '75-79', '80-84', '85+', '<1']

    # looping through each age within age groups
    for age_group in age_groups:
        # looping through each gender
        for gender in genders:
            # Creating and initialising the FB Prophet Model
            prophet = Prophet(
                daily_seasonality=False,
                yearly_seasonality=True,
                weekly_seasonality=False,
                changepoint_prior_scale=0.05,
                seasonality_prior_scale=10.0
            )

            # Creating a dataframe and filtering the current age group with the current gender.
            df_filtered = df_tbl2[(df_tbl2['Age group'] == age_group) & (df_tbl2['Sex'] == gender)]

            # Printing and displaying the actual number of deaths in each year for each age group within the UK.
            print(f"Actual number of deaths for Gender: {gender} and Age Group: {age_group}")
            print(df_filtered[['Year [note 3]', 'Number of deaths']].rename(columns={'Year [note 3]': 'Year'}))

            # Renaming and assigning my columns to specified column names to match with FB Prophet requirements and work
            # with the FB Prophet model successfully.
            df_filtered = df_filtered.rename(columns={'Year [note 3]': 'ds', 'Number of deaths': 'y'})

            # Convert 'ds' which represents the date column to datetime so that it can be read by the fb prophet model.
            df_filtered['ds'] = pd.to_datetime(df_filtered['ds'].astype(str), format='%Y')

            # Fitting the model onto my dataframe containing my dataset.
            prophet.fit(df_filtered)

            # Creating a dataframe to be used for the prediction forecasting
            future = prophet.make_future_dataframe(periods=10, freq='YE')

            # Passing the future Dataframe to generate a forecast prediction on deaths for the next 10 years for each
            # gender within each region within England.
            forecast = prophet.predict(future)

            # Rounding up or down the forecast values to the nearest whole number
            forecast['yhat'] = forecast['yhat'].round()

            # Plot the forecasts that were made by FB Prophet and display them to the user using plotly
            fig = plot_plotly(prophet, forecast)
            fig.update_layout(xaxis_title="Year",
                              yaxis_title="Number of Deaths",
                              title_text=f"FB Prophet prediction for Gender:{gender} and Age Group: {age_group} "
                              )
            fig.show()

            # Outputting and displaying the forecasts to the console.
            print(f"Forecasted Deaths for Gender: {gender} and Age Group: {age_group}")
            print(forecast[['ds', 'yhat']].rename(columns={'ds': 'Date', 'yhat': 'Number of Deaths'}))
            print("___________________________________________________________________________________________________")

model_forecast_deaths_for_age_groups_tbl2()