import pandas as pd
from src.data_loader import AmesDataLoader


def test_initial_clean_drops_id_and_normalizes_sentinals():
    df = pd.DataFrame({
        "Id": [1, 2],
        "SalePrice": [200000, 300000],
        "Alley": [None, "Grvl"],
        "GrLivArea": [1200, 1500],
    })
    loader = AmesDataLoader()
    cleaned = loader.initial_clean(df)
    assert "Id" not in cleaned.columns
    # Alley should be filled with the sentinel string 'None' for NA-means-none columns
    assert cleaned.loc[0, "Alley"] == "None"
    # SalePrice rows should remain
    assert cleaned["SalePrice"].min() > 0
