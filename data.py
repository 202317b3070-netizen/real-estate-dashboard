from sklearn.datasets import fetch_california_housing
 
def load_data():
    data = fetch_california_housing(as_frame=True)
    df = data.frame
    df['Price'] = data.target * 100000
    return df
 
