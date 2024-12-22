from sklearn.linear_model import LogisticRegression
from .preprocessing import normalize_data


def train_model(x,y):
    x_norm = normalize_data(x)
    model = LogisticRegression()
    return model.fit (x_norm, y)

