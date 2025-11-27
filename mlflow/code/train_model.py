import ray
import mlflow
import mlflow.sklearn
from sklearn.linear_model import LinearRegression
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

@ray.remote
def train_model():
    # Generate sample data
    X, y = make_regression(n_samples=100, n_features=1, noise=10, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Start MLflow run
    with mlflow.start_run():
        # Train model
        model = LinearRegression()
        model.fit(X_train, y_train)
        
        # Make predictions
        y_pred = model.predict(X_test)
        mse = mean_squared_error(y_test, y_pred)
        
        # Log metrics and model
        mlflow.log_metric("mse", mse)
        mlflow.sklearn.log_model(model, "linear_regression_model")
        
        return mse

if __name__ == "__main__":
    ray.init()
    result = ray.get(train_model.remote())
    print(f"Model trained with MSE: {result}")
    ray.shutdown()