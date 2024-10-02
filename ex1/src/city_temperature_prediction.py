import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import polynomial_fitting as pf

FIGSIZE = (10, 6)

DIGIT_NUM_ROUND = 2

COL_TO_REMOVE = ['Temp', 'Date', 'Country', 'City', 'Year', 'Month', 'Day']


def preprocess_data(data: pd.DataFrame) -> pd.DataFrame:
    data = data.dropna()

    data = data[data['Temp'] > 0]
    data = data[data['Month'].isin(range(1, 13))]
    data = data[data['Day'].isin((range(1, 32)))]

    # check that the date is equals to the columns year, month, day
    data = data[data['Date'].dt.year == data['Year']]
    data = data[data['Date'].dt.month == data['Month']]
    data = data[data['Date'].dt.day == data['Day']]

    data = data[data['DayOfYear'].isin(range(1, 366))]
    return data


def load_data(filename: str) -> pd.DataFrame:
    """
    Load city daily temperature dataset and preprocess data.
    Parameters
    ----------
    filename: str
        Path to house prices dataset

    Returns
    -------
    Design matrix and response vector (Temp)
    """

    # check exception for file not found
    try:

        df = pd.read_csv(filename, parse_dates=['Date'])
        df['DayOfYear'] = df['Date'].dt.dayofyear
        df = preprocess_data(df)

        return df


    except FileNotFoundError:
        print("File not found")
        return None


if __name__ == '__main__':
    # Question 2 - Load and preprocessing of city temperature dataset
    np.random.seed(0)
    df = load_data("city_temperature.csv")

    # Question 3 - Exploring data for specific country

    df_israel = df[df['Country'] == "Israel"]

    list_day_of_year = list(range(1, 366))

    plt.figure(figsize=(10, 6))
    min_year = min(df_israel['Year'])
    max_year = max(df_israel['Year'])
    for year in range(min_year, max_year + 1):
        df_year = df_israel[df_israel['Year'] == year]
        plt.scatter(df_year['DayOfYear'], df_year['Temp'], alpha=0.7, edgecolors='w',
                    label='Day Of Year {}'.format(year), s=20)

    plt.xlabel('Day of Year')
    plt.ylabel('Temperature')
    plt.title('Day of Year Per Temperature')
    plt.legend()
    plt.grid(True)
    plt.savefig('day_of_year_temperature.png')
    # plt.show()
    plt.close()

    ################################################################

    # 3B

    monthly_std = df_israel.groupby('Month')['Temp'].agg(['std']).reset_index()
    monthly_std.columns = ['Month', 'Temperature_STD']
    plt.figure(figsize=FIGSIZE)
    plt.bar(monthly_std['Month'], monthly_std['Temperature_STD'], color='skyblue')
    plt.xlabel('Month')
    plt.ylabel('Standard Deviation of Daily Temperatures')
    plt.title('Standard Deviation of Daily Temperatures by Month')
    plt.xticks(monthly_std['Month'])
    plt.grid(True)
    plt.savefig('groupby_israel_month_bar_plot.png')
    # plt.show()
    plt.close()

    # new praph.

    monthly_std = df.groupby(['Country', 'Month'])['Temp'].agg(['mean', 'std']).reset_index()
    monthly_std.columns = ['Country', 'Month', 'Temperature_Mean', 'Temperature_STD']
    plt.figure(figsize=(10, 6))
    for country in monthly_std['Country'].unique():
        country_data = monthly_std[monthly_std['Country'] == country]
        plt.errorbar(country_data['Month'], country_data['Temperature_Mean'], yerr=country_data['Temperature_STD'],
                     label=country, capsize=5, marker='o', linestyle='-', capthick=2)
    plt.xlabel('Month')
    plt.ylabel('Average Temperature')
    plt.title('Average Monthly Temperature by Country with Standard Deviation Error Bars')
    plt.legend(title='Country')
    plt.grid(True)
    plt.savefig('average_monthly_temperature_by_country_with_error_bars.png')
    plt.close()
    # plt.show()

    # Question 5 - Fitting model for different values of `k`

    df_israel_after_preprocess = preprocess_data(df_israel)
    X, y = df_israel.DayOfYear, df_israel.Temp

    # Randomly split the dataset into a training set (75%) and test set (25%).
    train_indices = X.sample(frac=0.75, random_state=0).index
    #
    X_train = X.loc[train_indices]
    # # print(X_train)
    y_train = y.loc[train_indices]

    # # The remaining 25% of the data for testing
    X_test = X.drop(train_indices)
    y_test = y.drop(train_indices)

    # # For every value k ∈ [1,10], fit a polynomial model of degree k using the training set.

    loss_list = list()

    for k in range(1, 11):
        fitted_model = pf.PolynomialFitting(k)
        fitted_model.fit(X_train, y_train)
        res_predict = fitted_model.predict(X_test)

        loss = fitted_model.loss(X_test, y_test)

        loss_list.append(round(loss, DIGIT_NUM_ROUND))
        print("for k: ", k, " test error is: ", round(loss, DIGIT_NUM_ROUND))

    plt.figure(figsize=FIGSIZE)
    plt.bar(list(range(1, 11)), loss_list, color='skyblue')
    plt.xlabel('test errorvalue of k')
    plt.ylabel('test error')
    plt.title('test error recorded for each value of k')
    plt.ylim(min(loss_list) - 10, max(loss_list) + 10)
    plt.xticks(list(range(1, 11)))
    plt.grid(True)
    plt.savefig('test_error_recorded_for_each_value_of_k.png')
    # plt.show()
    plt.close()

    # Question 6 - Evaluating fitted model on different countries

    # the minimal k: loss_list.index(min(loss_list))+1

    df_after_preprocess = preprocess_data(df)

    fitted_model = pf.PolynomialFitting(loss_list.index(min(loss_list)) + 1)
    df_israel_after_preprocess = preprocess_data(df_israel)
    X, y = df_israel.DayOfYear, df_israel.Temp
    fitted_model.fit(X, y)
    res_predict = fitted_model.predict(X)
    loss_dict = dict()

    for country in set(df['Country']):
        df_country = df_after_preprocess[df_after_preprocess['Country'] == country]
        X, y = df_country.DayOfYear, df_country.Temp

        loss = fitted_model.loss(X, y)

        loss_dict[country] = round(loss, DIGIT_NUM_ROUND)

    plt.figure(figsize=FIGSIZE)
    plt.bar(list(loss_dict.keys()), list(loss_dict.values()), color='skyblue')

    plt.xlabel('Country')
    plt.ylabel('the Model\'s Error')
    plt.title('by chosen k model error by country.png')
    plt.grid(True)
    plt.savefig('by_chosen_k_model_error_by_country.png')
    # plt.show()
    plt.close()
