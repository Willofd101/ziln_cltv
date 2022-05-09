import os
import numpy as np
import pandas as pd
import tqdm

## Function to read transaction.csv file
def load_data(company):
    print("Running load_data() for: ", company)
    all_data_filename = 'data/transactions.csv'
    one_company_data_filename = (
      'data/transactions_company_{}.csv'
      .format(company))
    if os.path.isfile(one_company_data_filename):
        df = pd.read_csv(one_company_data_filename)
    else:
        data_list = []
        chunksize = 10**6
        # 350 iterations
        for chunk in tqdm.tqdm(pd.read_csv(all_data_filename, chunksize=chunksize)):
            data_list.append(chunk.query("company=={}".format(company)))
        df = pd.concat(data_list, axis=0)
        df.to_csv(one_company_data_filename, index=None)
    return df

##Function to preprocess the data
def preprocess(df):
    df = df.query('purchaseamount>0')
    df['date'] = pd.to_datetime(df['date'], format='%Y-%m-%d')
    df['start_date'] = df.groupby('id')['date'].transform('min')

    # Compute calibration values
    calibration_value = (
      df.query('date==start_date').groupby('id')
      ['purchaseamount'].sum().reset_index())
    calibration_value.columns = ['id', 'calibration_value']

    # Compute holdout values
    one_year_holdout_window_mask = (
      (df['date'] > df['start_date']) &
      (df['date'] <= df['start_date'] + np.timedelta64(365, 'D')))
    holdout_value = (
      df[one_year_holdout_window_mask].groupby('id')
      ['purchaseamount'].sum().reset_index())
    holdout_value.columns = ['id', 'holdout_value']

    # Compute calibration attributes
    calibration_attributes = (
      df.query('date==start_date').sort_values(
          'purchaseamount', ascending=False).groupby('id')[[
              'chain', 'dept', 'category', 'brand', 'productmeasure'
          ]].first().reset_index())

    # Merge dataframes
    customer_level_data = (
      calibration_value.merge(calibration_attributes, how='left',
                              on='id').merge(
                                  holdout_value, how='left', on='id'))
    customer_level_data['holdout_value'] = (
      customer_level_data['holdout_value'].fillna(0.))
    categorical_features = ([
      'chain', 'dept', 'category', 'brand', 'productmeasure'
    ])
    customer_level_data[categorical_features] = (
      customer_level_data[categorical_features].fillna('UNKNOWN'))

    # Specify data types
    customer_level_data['log_calibration_value'] = (
      np.log(customer_level_data['calibration_value']).astype('float32'))
    customer_level_data['chain'] = (
      customer_level_data['chain'].astype('category'))
    customer_level_data['dept'] = (customer_level_data['dept'].astype('category'))
    customer_level_data['brand'] = (
      customer_level_data['brand'].astype('category'))
    customer_level_data['category'] = (
      customer_level_data['category'].astype('category'))
    customer_level_data['label'] = (
      customer_level_data['holdout_value'].astype('float32'))
    return customer_level_data


def process(company):
    print("Process company {}".format(company))
    transaction_level_data = load_data(company)
    customer_level_data = preprocess(transaction_level_data)
    customer_level_data_file = (
      "data/customer_level_data_company_{}.csv"
      .format(company))
    customer_level_data.to_csv(customer_level_data_file, index=None)
    print("Done company {}".format(company))
