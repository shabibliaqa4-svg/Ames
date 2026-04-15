import pandas as pd
from src.feature_engineering import FeatureEngineer


def test_feature_engineer_basic():
    df = pd.DataFrame({
        'GrLivArea':[1000,1500],
        'TotalBsmtSF':[500,600],
        '1stFlrSF':[800,900],
        '2ndFlrSF':[200,100],
        'FullBath':[2,3],
        'HalfBath':[1,0],
        'BsmtFullBath':[1,1],
        'BsmtHalfBath':[0,1],
        'YearBuilt':[1980,1990],
        'YrSold':[2010,2011],
        'GarageArea':[0,200],
        'GarageCars':[0,1],
        'TotRmsAbvGrd':[5,6],
        'MasVnrArea':[0,100],
        'MoSold':[6,7]
    })
    fe = FeatureEngineer()
    df2 = fe.transform(df)
    assert 'TotalSF' in df2.columns
    assert 'TotalBathrooms' in df2.columns

