from src.easy_task.feature_engineering import get_features
from sklearn.preprocessing import StandardScaler

def get_processed_features(features):
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(features)
    print("Scaled Features mean, std:", scaled_features.mean().round(4), scaled_features.std().round(4))
    return scaled_features
if __name__=="__main__":
    features,genres,paths=get_features()
    scaled_features=get_processed_features(features)
    print("Scaled Features Shape: ",scaled_features.shape)
