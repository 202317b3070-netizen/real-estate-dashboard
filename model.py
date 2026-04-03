from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
 
def train_model(X, y, model_type):
    if model_type == "Linear Regression":
        model = LinearRegression()
    else:
        model = RandomForestRegressor()
 
    model.fit(X, y)
    return model
 
def predict(model, features):
    return model.predict([features])[0]
