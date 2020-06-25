"""

	Simple Streamlit webserver application for serving developed classification
	models.

	Author: Explore Data Science Academy.

	Note:
	---------------------------------------------------------------------
	Please follow the instructions provided within the README.md file
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

# Data dependencies
import warnings
warnings.simplefilter(action='ignore')

# Prerequisites
import subprocess
import sys
#pip install wordcloud scikit-learn bs4 lmxl

# Load Dependencies
import re
import pickle
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from PIL import Image
from bs4 import BeautifulSoup
from collections import Counter
from wordcloud import WordCloud
from sklearn.svm import LinearSVC
from nltk.corpus import stopwords
from sklearn.utils import resample
from sklearn.pipeline import Pipeline
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer, TfidfTransformer
from sklearn.metrics import f1_score, accuracy_score, classification_report, confusion_matrix

# Display
sns.set(font_scale=1)
sns.set_style("white")

# Vectorizer
tweet_vectorizer = open("resources/test_vect2.pkl","rb")
tweet_cv = joblib.load(tweet_vectorizer) # loading vectorizer from the pkl file

# Load your raw data
raw = pd.read_csv("Data/train.csv")

# Data cleaning
stop_words = stopwords.words('english')
stop_words.extend(['via','rt'])
def remove_noise(tweet):
	"""
	Remove noise from text data, such as newlines, punctuation, URLs and numbers.
	"""
	new_tweet = BeautifulSoup(tweet, "html.parser").text #HTML Decoding
	new_tweet = re.sub(r'http\S+', 'urlweb', tweet) #Replace URLs
	new_tweet = new_tweet.lower()
	new_tweet = new_tweet.replace('\n',' ') #Remove Newlines
	new_tweet = re.sub('[/(){}\[\]\|@,;]',' ', new_tweet)
	new_tweet = re.sub('[^0-9a-z #+_]', '', new_tweet) #Remove Symbols
	new_tweet = ' '.join(word for word in new_tweet.split() if word not in stop_words)
	return new_tweet

# Lemmatisation
lemmatiser = WordNetLemmatizer()
def lemmatise_words(text):
    words = []
    for word in text.split():
        words.append(lemmatiser.lemmatize(word))
    return " ".join(words)

# The main function where we will build the actual app
def main():
	"""Tweet Classifier App with Streamlit """
	# Creates a main title and subheader on your page -
	# these are static across all pages
	#st.title("Tweet Classification")
	st.subheader("Team RM5 SigmoidFreuds")
	image = Image.open('rm5_banner.png')
	st.image(image,use_column_width=True)
	# Creating sidebar with selection box -
	# you can create multiple pages this way
	options = ["Tweet Sentiment Prediction", "Exploratory Data Analysis", "About This Project"]
	selection = st.sidebar.selectbox("Select Page", options)
	st.sidebar.subheader("Page Descriptions:")
	st.sidebar.markdown('**Tweet Sentiment Prediction** - Predicts tweet sentiment. Obviously.')
	st.sidebar.markdown('**Exploratory Data Analysis** - Explores the data and stuff.')
	st.sidebar.markdown('**About This Project** - Some useless information.')

	# Building out the "About This Project" page
	if selection == "About This Project":
		st.info("Some useless information.")
		# You can read a markdown file from supporting resources folder
		st.markdown("Some information here")

		st.subheader("Raw Twitter data and label")
		if st.checkbox('Show raw data'): # data is hidden if box is unchecked
			st.write(raw[['sentiment', 'message']]) # will write the df to the page

	# Building out the "Exploratory Data Analysis" page
	if selection == "Exploratory Data Analysis":
		st.info("Explores the data and stuff.")
		# You can read a markdown file from supporting resources folder
		if st.checkbox("Preview Raw Data"): # data is hidden if box is unchecked
			if st.button("Head"):
				st.write(raw.head())
			if st.button("Tail"):
				st.write(raw.tail())

		data_dim = st.radio('View Dataset Dimensions',('Rows','Columns'))
		if data_dim == 'Rows':
			st.text("Number of Rows in Data")
			st.write(len(raw))
		if data_dim == 'Columns':
			st.text("Number of Columns in Data")
			st.write(raw.shape[1])

		st.subheader("Sentiment Classes")
		# Number of Tweets Per Sentiment Class
		fig, axis = plt.subplots(ncols=2, figsize=(10, 5))

		ax = sns.countplot(x='sentiment',data=raw,palette='winter',ax=axis[0])
		axis[0].set_title('Number of Tweets Per Sentiment Class',fontsize=14)
		axis[0].set_xlabel('Sentiment Class')
		axis[0].set_ylabel('Tweets')
		for p in ax.patches:
			ax.text(p.get_x() + p.get_width()/2., p.get_height(), '%d' % int(p.get_height()), fontsize=11, ha='center', va='bottom')

		raw['sentiment'].value_counts().plot.pie(autopct='%1.1f%%',colormap='winter_r',ax=axis[1])
		axis[1].set_title('Proportion of Tweets Per Sentiment Class',fontsize=14)
		axis[1].set_ylabel('Sentiment Class')

		st.pyplot()

	# Building out the "Tweet Sentiment Prediction" page
	if selection == "Tweet Sentiment Prediction":
		st.info("Predicts tweet sentiment. Obviously.")

		st.markdown(' ')
		st.markdown('**Class Descriptions:**')
		st.markdown('**News:** The tweet links to factual news about climate change')
		st.markdown('**Pro:** The tweet supports the belief of man-made climate change')
		st.markdown('**Neutral:** The tweet neither supports nor refutes the belief of man-made climate change')
		st.markdown('**Anti:** The tweet does not believe in man-made climate change')
		st.markdown(' ')

		# Creating a text box for user input
		tweet_text = st.text_area("Enter Text Below","")
		tweet_text = remove_noise(tweet_text)
		tweet_text = lemmatise_words(tweet_text)

		all_ml_models = ['Linear SVC','Logistic Regression','Multinomial Naive Bayes', 'Random Forest Classifier','K Neighbours Classifier','Decision Tree Classifier','AdaBoost Classifier']
		model_choice = st.selectbox("Select Model",all_ml_models)

		prediction_labels = {'business': 0,'tech': 1,'sport': 2,'health': 3,'politics': 4,'entertainment': 5}
		if st.button("Classify"):
			# Transforming user input with vectorizer
			vect_text = tweet_cv.transform([tweet_text]).toarray()
			# Load your .pkl file with the model of your choice + make predictions
			# Try loading in multiple models to give the user a choice
			if model_choice == 'Linear SVC':
				predictor = joblib.load(open(os.path.join("resources/svc_model.pkl"),"rb"))
				prediction = predictor.predict(vect_text)
			elif model_choice == 'Logistic Regression':
				predictor = joblib.load(open(os.path.join("resources/logreg_model.pkl"),"rb"))
				prediction = predictor.predict(vect_text)
			elif model_choice == 'Multinomial Naive Bayes':
				predictor = joblib.load(open(os.path.join("resources/multinb_model.pkl"),"rb"))
				prediction = predictor.predict(vect_text)
			elif model_choice == 'Random Forest Classifier':
				predictor = joblib.load(open(os.path.join("resources/rf_model.pkl"),"rb"))
				prediction = predictor.predict(vect_text)
			elif model_choice == 'K Neighbours Classifier':
				predictor = joblib.load(open(os.path.join("resources/kn_model.pkl"),"rb"))
				prediction = predictor.predict(vect_text)
			elif model_choice == 'Decision Tree Classifier':
				predictor = joblib.load(open(os.path.join("resources/dt_model.pkl"),"rb"))
				prediction = predictor.predict(vect_text)
			elif model_choice == 'AdaBoost Classifier':
				predictor = joblib.load(open(os.path.join("resources/ad_model.pkl"),"rb"))
				prediction = predictor.predict(vect_text)

			# When model has successfully run, will print prediction
			# You can use a dictionary or similar structure to make this output
			# more human interpretable.
			labels = {'News': 2,'Pro': 1,'Neutral': 0,'Anti': -1}
			for key,value in labels.items():
				if prediction == value:
					result = key
			st.success("Predicted Tweet Sentiment Class: {}".format(result))

# Required to let Streamlit instantiate our web app.
if __name__ == '__main__':
	main()
