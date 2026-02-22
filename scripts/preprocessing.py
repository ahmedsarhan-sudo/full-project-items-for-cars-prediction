import numpy as np 
import pandas as pd

def replace_categorical_by_numerical(df):
    df['Levy'].replace('-',0 , inplace= True)
    df['Levy'] = df.Levy.astype(int)
    clean_Engine = df['Engine volume'].str.extract(r'(\d*\.\d|\d)')
    df['Engine volume'] = clean_Engine.astype('float')
    df['Leather interior'] = df['Leather interior'].map({'Yes' : 1 ,'No' : 0})
    df['Mileage'] = df['Mileage'].str.extract(r'(\d+)').astype('int')

    return df


def fix_datatype(df):
    for col in df.columns:
        if df[col].dtype == 'object':
            df[col] = df[col].astype('string')
    return df


def columns_transformation(df):
    df['Levy_logp1'] = np.log1p(df['Levy'])
    df['Engine_volume_logp1'] = np.log1p(df['Engine volume'])
    df['Mileage_logp1'] = np.log1p(df['Mileage'])

    return df


def clean_outliers(df , cols):
    for col in cols:
        q1 = df[col].quantile(.25)
        q3 = df[col].quantile(.75)
        iqr = q3 - q1

        lower_bound = q1 - 1.5*iqr
        upper_bound = q3 + 1.5*iqr

        df = df[(df[col] >= lower_bound) & (df[col] <= upper_bound)]

    return df


def engineer_features(df):
    current_year = pd.Timestamp.now().year
    df['Age'] = current_year - df['Prod. year']
    
    return df


def preprocessing_pipline(df:pd.DataFrame):
    print(f'preprocessing started...')
    print(f'initial shape : {df.shape}')

    df.drop_duplicates(inplace=True)
    print(f'After dropping dublicates : {df.shape}')

    print(f'Replacing categorical values...')
    df = replace_categorical_by_numerical(df)

    print(f'fix all columns data type...')
    df = fix_datatype(df)

    df = clean_outliers(df , ['Price','Levy','Engine volume','Mileage'])
    print(f'After cleaning outlairs : {df.shape}')

    print(f'Doing columns transformation...')
    df = columns_transformation(df)

    print(f'Feature engineering...')
    df = engineer_features(df)

    
    print(f'Filtering data by importand manufacturer...')

    df = df[(df['Prod. year'] > 1991) & (df['Prod. year'] <2020)]

    filter_manufacturer_by_price = df.groupby('Manufacturer')['Price'].sum().sort_values(ascending=False).reset_index().iloc[36:]['Manufacturer'].values
    filter_manufacturer_by_count = df['Manufacturer'].value_counts().reset_index().iloc[31:]['Manufacturer'].values

    union_uninterseted = [i for i in filter_manufacturer_by_count if i for x in filter_manufacturer_by_price if i == x]

    df = df[~(df['Manufacturer'].isin(union_uninterseted))]

    print(f'After filtering shape : {df.shape}')

    print(f'Dropping columns...')
    df =df.drop(columns=['ID','Prod. year','Doors'])

    return df


