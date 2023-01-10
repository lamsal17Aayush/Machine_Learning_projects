import joblib

def predict(data):
    clf=joblib.load('/home/lamsal/Documents/ML/StreamlitTutorial/output_models/rf_model.sav')
    return clf.predict(data)