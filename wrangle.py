import os

import pandas as pd

from env import host, password, user


def get_connection(db, user=user, host=host, password=password):
    '''
    This function uses my info from my env file to
    create a connection url to access the Codeup db.
    '''
    return f'mysql+pymysql://{user}:{password}@{host}/{db}'


def get_zillow_data():
    """Seeks to read the cached zillow.csv first, if not found it will  """
    filename = "zillow.csv"

    if os.path.isfile(filename):
        return pd.read_csv(filename)
    else:
        return get_new_zillow_data()


def get_new_zillow_data():
    """Returns a dataframe of all 2017 properties that are Single Family Residential"""

    sql = """
    select
    bedroomcnt, bathroomcnt, calculatedfinishedsquarefeet, taxvaluedollarcnt, yearbuilt, taxamount, fips
    from properties_2017
    join propertylandusetype using (propertylandusetypeid)
    where propertylandusedesc = "Single Family Residential"
    """
    return pd.read_sql(sql, get_connection("zillow"))


def handle_nulls(df):
    # We keep 99.41% of the data after dropping nulls
    # round(df.dropna().shape[0] / df.shape[0], 4) returned .9941
    df = df.dropna()
    return df


def optimize_types(df):
    # Convert some columns to integers
    # fips, yearbuilt, and bedrooms can be integers
    df["fips"] = df["fips"].astype(int)
    df["yearbuilt"] = df["yearbuilt"].astype(int)
    df["bathroomcnt"] = df["bathroomcnt"].astype(int)
    df["taxamount"] = df["taxamount"].astype(int)
    df["bedroomcnt"] = df["bedroomcnt"].astype(int)
    df["taxvaluedollarcnt"] = df["taxvaluedollarcnt"].astype(int)
    df["calculatedfinishedsquarefeet"] = df["calculatedfinishedsquarefeet"].astype(
        int)
    return df


def handle_outliers(df):
    """Manually handle outliers that do not represent properties likely for 99% of buyers and zillow visitors
    so it gets rid of bathroom and bedroom counts more than six and properties valued more than 2,000,000"""
    df = df[df.bathroomcnt <= 6]

    df = df[df.bedroomcnt <= 6]

    df = df[df.taxvaluedollarcnt < 2_000_000]

    return df


def wrangle_zillow():
    """
    Acquires Zillow data from the server, handles null-values, optimizes the datatypes for machine learning, handles outliers, renames columns to something that is more legible, returns a clean dataframe and stores the dataframe locally
    """
    df = get_zillow_data()

    df = handle_nulls(df)

    df = optimize_types(df)

    df = handle_outliers(df)

    df = df.rename(columns={'bedroomcnt': 'bedrooms',
                            'bathroomcnt': 'bathrooms',
                            'calculatedfinishedsquarefeet': 'area',
                            'taxvaluedollarcnt': 'tax_value',
                            'yearbuilt': 'year_built',
                            'taxamount': 'tax_amount'})

    df.to_csv("zillow.csv", index=False)

    return df
