import streamlit as st
import pickle
import string
from nltk.corpus import stopwords
import nltk
from nltk.stem import WordNetLemmatizer
lemma = WordNetLemmatizer()

def transform_text(text):
    text = text.lower()
    text = nltk.word_tokenize(text)
    Q = []

    for i in text:
        if i.isalnum():
            Q.append(i)
    text = Q[:]
    Q.clear()

    for i in text:
        if i not in stopwords.words('english') and i not in string.punctuation:
            Q.append(i)
    text = Q[:]
    Q.clear()

    for i in text:
        Q.append(lemma.lemmatize(i))
    return " ".join(Q)

model1 = pickle.load(open('vectorizer.pkl','rb'))
model2 = pickle.load(open('model.pkl','rb'))

st.title("SMS Spam Classifier")
input_sms = st.text_area("Enter The Message")

if st.button('Predict'):
    transformed_sms = transform_text(input_sms)
    vector_input = model1.transform([transformed_sms])
    result = model2.predict(vector_input)[0]

    if result == 1:
        st.header("Spam")
    else:
        st.header("Not Spam")