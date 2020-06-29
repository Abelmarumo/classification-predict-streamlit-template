import streamlit as st
import joblib,os
import spacy
from wordcloud import WordCloud
from PIL import Image
import matplotlib.pyplot as plt

nlp = spacy.load('en')

multinomial_logistic = joblib.load(open(os.path.join("resources/multinomial_logistic.plk"),"rb"))
LinearSVC = joblib.load(open(os.path.join("resources/LinearSVC.plk"),"rb"))


def get_keys(value,my_dic):
    for key,val in my_dic.items():
        if val==value:
            return key
def main():
    """tweet classifier"""
    st.title("Tweet Classifier")
    st.subheader("NLP and ML App with streamlit")

    activities =["Prediction","NLP"]
    choice = st.sidebar.selectbox("Choose Activity",activities)

    if choice=='Prediction':
        st.info("Prediction with ML")
        tweet_text = st.text_area("Enter Text","Type Here")
        all_ml_modules = ["MLR","LSVC"]
        model_choice = st.selectbox("Choose the ML model",all_ml_modules)
        labels = {'News':2,'Pro':1,'Neutral':0,'Anti':-1}
        if st.button("Classify"):
            st.text("Original tweet ::\n{}".format(tweet_text))
            if model_choice=="MLR":
                # predictor = joblib.load(open(os.path.join("resources/multinomial_logistic.plk"),"rb"))
                prediction = multinomial_logistic.predict([tweet_text])
                ##st.write(prediction[0])
                st.success(f"The tweet is categorized as {get_keys(prediction[0],labels)}")
            if model_choice=="LSVC":
                # predictor = joblib.load(open(os.path.join("resources/LinearSVC.plk"),"rb"))
                prediction = LinearSVC.predict([tweet_text])
                ##st.write(prediction[0])
                st.success(f"The tweet is categorized as {get_keys(prediction[0],labels)}")

    if choice=='NLP':
        st.info("Natural Language Processing")
        tweet_text = st.text_area("Enter Text","Type Here")
        nlp_task = ["Tokenization","NER","Lemmatization","POS"]
        task_choice = st.selectbox("Choose NLP Task",nlp_task)
        if st.button('Analysis'):
            st.info("Original tweet ::\n{}".format(tweet_text))

            doc = nlp(tweet_text)
            if task_choice =="Tokenization":
                tokens = [t.text for t in doc]

            elif task_choice =="NER":
                tokens = [(ent.text,ent.label_) for ent in doc.ents]

            elif task_choice =="Lemmatization":
                tokens = [f"'Token':{t.text},'Lemma':{t.lemma_}" for t in doc]
            elif task_choice =="POS":
                tokens = [f"'Token':{t.text},'POS':{t.pos_},'Dependency':{t.dep_}" for t in doc]
            st.json(tokens)
        if st.button("Wordcloud"):
            wordcloud  = WordCloud().generate(tweet_text)
            plt.imshow(wordcloud)
            plt.axis('off')
            st.pyplot()



if __name__ == "__main__":
    main()