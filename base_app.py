"""

    Simple Streamlit webserver application for serving developed classification
	models.

    Author: Explore Data Science Academy.

    Note:
    ---------------------------------------------------------------------
    Plase follow the instructions provided within the README.md file
    located within this directory for guidance on how to use this script
    correctly.
    ---------------------------------------------------------------------

    Description: This file is used to launch a minimal streamlit web
	application. You are expected to extend the functionality of this script
	as part of your predict project.

	For further help with the Streamlit framework, see:

	https://docs.streamlit.io/en/latest/

"""
# Streamlit dependencies
import streamlit as st
import joblib,os
import numpy as np
import matplotlib.pyplot as plt

# Data dependencies
import pandas as pd

# Vectorizer
news_vectorizer = open("resources/tfidfvect.pkl","rb")
tweet_cv = joblib.load(news_vectorizer) # loading your vectorizer from the pkl file

# Load your raw data
raw = pd.read_csv("resources/train.csv")

# The main function where we will build the actual app
def main():
	"""Tweet Classifier App with Streamlit """

	# Creates a main title and subheader on your page -
	# these are static across all pages
	st.title("Tweet Classifer")
	st.subheader("Climate change tweet classification")

	# Creating sidebar with selection box -
	# you can create multiple pages this way
	options = ["Classifier Page", "General Information Page"]
	selection = st.sidebar.selectbox("Choose Page", options)

	
	
	
	# Building out the General inforation page
	if selection=="General Information Page":
		sub_options1= ["Bar graph", "Pie chart"]
		sub_selection1 = st.sidebar.selectbox("General information", sub_options1)
		if sub_selection1 == "Bar graph":
			raw['sentiment'].value_counts().plot(kind='bar')
			st.pyplot()
			st.info("General Information")
			# You can read a markdown file from supporting resources folder
			st.markdown("Some information here")

			st.subheader("Raw Twitter data and label")
			if st.checkbox('Show raw data'): # data is hidden if box is unchecked
				st.write(raw[['sentiment', 'message']]) # will write the df to the page
		else:
			plt.pie(x=raw['sentiment'].value_counts().values,labels=raw['sentiment'].value_counts().index)
			st.pyplot()
			st.info("General Information")
			# You can read a markdown file from supporting resources folder
			st.markdown("Some information here")

			st.subheader("Raw Twitter data and label")
			if st.checkbox('Show raw data'): # data is hidden if box is unchecked
				st.write(raw[['sentiment', 'message']]) # will write the df to the page


	if selection=="Classifier Page":
		sub_options2= ["Multinomial Logistic Regression", "LinearSVC"]
		sub_selection2 = st.sidebar.selectbox("Choose classifier", sub_options2)
		if sub_selection2 == "Multinomial Logistic Regression":
			st.info("Prediction with "+sub_selection2)
			# Creating a text box for user input
			tweet_text = st.text_area("Enter Text","Type Here")

			if st.button("Classify"):
				# Transforming user input with vectorizer
				#vect_text = tweet_cv.transform([tweet_text]).toarray()
				# Load your .pkl file with the model of your choice + make predictions
				# Try loading in multiple models to give the user a choice
				predictor = joblib.load(open(os.path.join("resources/multinomial_logistic.plk"),"rb"))
				prediction = predictor.predict([tweet_text])

				# When model has successfully run, will print prediction
				# You can use a dictionary or similar structure to make this output
				# more human interpretable.
				if prediction[0]==2:
					st.success("Text Categorized as News")
				elif prediction[0]==1:	
					st.success("Text Categorized as Pro")
				elif prediction[0]==0:	
					st.success("Text Categorized as Neutral")
				elif prediction[0]==-1:	
					st.success("Text Categorized as Anti")
###
		if sub_selection2 == "LinearSVC":
			st.info("Prediction with "+sub_selection2)
			# Creating a text box for user input
			tweet_text = st.text_area("Enter Text","Type Here")

			if st.button("Classify"):
				# Transforming user input with vectorizer
				#vect_text = tweet_cv.transform([tweet_text]).toarray()
				# Load your .pkl file with the model of your choice + make predictions
				# Try loading in multiple models to give the user a choice
				predictor = joblib.load(open(os.path.join("resources/LinearSVC.plk"),"rb"))
				prediction = predictor.predict([tweet_text])

				# When model has successfully run, will print prediction
				# You can use a dictionary or similar structure to make this output
				# more human interpretable.
				if prediction[0]==2:
					st.success("Text Categorized as News")
				elif prediction[0]==1:	
					st.success("Text Categorized as Pro")
				elif prediction[0]==0:	
					st.success("Text Categorized as Neutral")
				elif prediction[0]==-1:	
					st.success("Text Categorized as Anti")
	
###
# Required to let Streamlit instantiate our web app.  
if __name__ == '__main__':
	main()
