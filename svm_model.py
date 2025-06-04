from sklearn.svm import SVR
from sklearn.model_selection import train_test_split

def split_data(feature, target, test_size):
    x_train, x_test, y_train, y_test = train_test_split(feature, target, test_size=test_size)
    return x_train, x_test, y_train, y_test

def train_model(kernel, c, x_train, y_train):
    svr = SVR(kernel=kernel, C=c)
    svr.fit(x_train, y_train)
    return svr

def calculate_accuracy(svr, x_test, y_test):
    return svr.score(x_test, y_test)

def predict(model, x_future):
    return model.predict(x_future)
