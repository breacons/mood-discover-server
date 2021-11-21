import joblib

rf = joblib.load('./model/hatespeech-v2.pkl')
tfidf_vectorizer = joblib.load('./model/vectorizer.pkl')


# 0 - hate speech 1 - offensive language 2 - neither
def predict_hate_speech(message):
    transformed = tfidf_vectorizer.transform([message])
    result = rf.predict(transformed)[0]

    if result == 0:
        return 'HATE_SPEECH'

    if result == 1:
        return 'OFFENSIVE_LANGUAGE'

    return 'NEITHER'
