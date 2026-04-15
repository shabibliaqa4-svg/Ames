import pandas as pd
from src.preprocessing import build_preprocessor


def test_build_preprocessor_transforms_numeric_and_categorical():
    df = pd.DataFrame({
        "LotArea": [1000, 2000],
        "GrLivArea": [800, 1200],
        "MSSubClass": ["20", "60"],
        "MSZoning": ["RL", "RM"],
    })
    preproc = build_preprocessor(df)
    # Should be able to fit and transform without errors
    X = preproc.fit_transform(df)
    assert X.shape[0] == df.shape[0]
