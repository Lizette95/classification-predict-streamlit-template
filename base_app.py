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

# Ignore warnings
import warnings
warnings.simplefilter(action='ignore')

# Install Prerequisites
# import sys
# import nltk
# !{sys.executable} -m pip install bs4 lxml wordcloud scikit-learn scikit-plot

# Exploratory Data Analysis
import re
import ast
import time
import nltk
import pickle
import numpy as np
import pandas as pd
import seaborn as sns
from textblob import TextBlob
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from nltk.sentiment import SentimentIntensityAnalyzer


# Data Preprocessing
import string
from bs4 import BeautifulSoup
from collections import Counter
from nltk.corpus import wordnet
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import TweetTokenizer
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer

# Classification Models
from sklearn.svm import LinearSVC, SVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

# Performance Evaluation
from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import label_binarize
from sklearn.model_selection import GridSearchCV
from scikitplot.metrics import plot_roc, plot_confusion_matrix
from sklearn.multiclass import OneVsRestClassifier, OneVsOneClassifier
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score, classification_report, confusion_matrix

# Display
sns.set(font_scale=1)
sns.set_style("white")
from PIL import Image

# Vectorizer
tweet_vectorizer = open("resources/models/TfidfVectorizer.pkl","rb")
tweet_cv = joblib.load(tweet_vectorizer) # loading vectorizer from the pkl file

# Load your raw data
raw = pd.read_csv("Data/train.csv")
df_train = raw.copy()

# Data cleaning
def clean(tweet_text):
	token = (TweetTokenizer().tokenize(tweet_text)) ## first we tokenize
	punc = [i for i in token if i not in list(string.punctuation)] ## remove punctuations
	dig = [i for i in punc if i not in list(string.digits)] ## remove digits
	final = [i for i in dig if len(i) > 1] ## since we not removing stopwords, remove all words with only 1 character
	return final

# Lemmatisation
def get_part_of_speech(word):
	probable_part_of_speech = wordnet.synsets(word) ## finding word that is most similar (synonyms) for semantic reasoning
	pos_counts = Counter() # instantiating our counter class
	## finding part of speech of word if part of speech is either noun, verb, adjective etc and add it up in a list
	pos_counts["n"] = len([item for item in probable_part_of_speech if item.pos()=="n"])
	pos_counts["v"] = len([item for item in probable_part_of_speech if item.pos()=="v"])
	pos_counts["a"] = len([item for item in probable_part_of_speech if item.pos()=="a"])
	pos_counts["r"] = len([item for item in probable_part_of_speech if item.pos()=="r"])
	most_likely_part_of_speech = pos_counts.most_common(1)[0][0] ## will extract the most likely part of speech from the list
	return most_likely_part_of_speech

normalizer = WordNetLemmatizer()
def lemmatise_words(final):
	lemma = [normalizer.lemmatize(token, get_part_of_speech(token)) for token in final] ## lemmatize by way of applying part of speech
	return ' '.join(lemma)

def highlight(val):
	return'background-color: yellow'

labels = {'News': 2,'Pro': 1,'Neutral': 0,'Anti': -1}

# The main function where we will build the actual app
def main():
	"""Tweet Classifier App with Streamlit """
	# Creates a main title and subheader on your page -
	# these are static across all pages
	#st.title("Tweet Classification")
	st.subheader("Team RM5 SigmoidFreuds")
	banner = Image.open('resources/imgs/rm5_banner.png')
	st.image(banner,use_column_width=True)
	# Creating sidebar with selection box -
	# you can create multiple pages this way
	options = ["About This Project","Exploratory Data Analysis","Tweet Sentiment Prediction"]
	selection = st.sidebar.selectbox("Select a Page", options)

	# Building out the "About This Project" page
	if selection == "About This Project":
		st.sidebar.subheader("Page Descriptions:")
		st.sidebar.markdown(' ')
		st.sidebar.markdown('**About This Project** - Read about the application and the data that was used to solve the problem at hand.')
		st.sidebar.markdown('**Exploratory Data Analysis** - Explore and draw insights from labelled training data.')
		st.sidebar.markdown('**Tweet Sentiment Prediction** - Choose a trained classification model to predict tweet sentiment and evaluate model performance.')
		st.info("Read about the application and the data that was used to solve the problem at hand.")
		st.markdown('**Problem Statement:**')
		st.markdown('Build a classification model that is able to classify whether or not a person believes in climate change, based on their novel tweet data.')
		st.markdown('**Application:**')
		st.markdown('This application requires the user to input text (ideally a tweet relating to climate change), and will classify it according to whether or not they believe in climate change. Below you will find information about the data source and a brief data description. You can have a look at word clouds and other general EDA on the "Exploratory Data Analysis" page, and make your predictions on the "Tweet Sentiment Prediction" page that you can navigate to in the sidebar.')
		st.markdown('**Variables:**')
		st.markdown('* sentiment: Sentiment of tweet')
		st.markdown('* message: Tweet body')
		st.markdown('* tweetid: Unique Twitter ID')
		st.markdown('**Data Source:**')
		st.markdown('The collection of this data was funded by a Canada Foundation for Innovation JELF Grant to Chris Bauch, University of Waterloo. This dataset aggregates tweets pertaining to climate change collected between Apr 27, 2015 and Feb 21, 2018. In total, 43943 tweets were annotated. Each tweet is labelled independently by 3 reviewers. This dataset only contains tweets that all 3 reviewers agreed on (the rest were discarded).')
		st.markdown(' ')
		st.markdown('**Sentiment Class Descriptions:**')
		st.markdown('**News (2):** The tweet links to factual news about climate change')
		st.markdown('**Pro (1):** The tweet supports the belief of man-made climate change')
		st.markdown('**Neutral (0):** The tweet neither supports nor refutes the belief of man-made climate change')
		st.markdown('**Anti (-1):** The tweet does not believe in man-made climate change')

	# Building out the "Exploratory Data Analysis" page
	if selection == "Exploratory Data Analysis":
		st.info("Explore and draw insights from labelled training data.")
		st.sidebar.subheader("Sections:")
		data = st.sidebar.checkbox('Data Summary')
		wordclouds = st.sidebar.checkbox('Wordclouds')
		label = st.sidebar.checkbox('Label (Sentiment)')
		word_freq = st.sidebar.checkbox('Word Frequencies')
		urls = st.sidebar.checkbox('URLs')
		hashtags = st.sidebar.checkbox('Hashtags')
		mentions = st.sidebar.checkbox('Mentions')
		retweets = st.sidebar.checkbox('Retweets')

		if data:
			st.subheader("Data Summary")
			row_num = st.slider("Select Number of Rows to View", 1, 100, 10)
			if st.checkbox("Preview Raw Dataset"): # data is hidden if box is unchecked
				st.write(df_train.head(row_num))
			data_dim = st.radio('View Dataset Dimensions',('Rows','Columns'))
			if data_dim == 'Rows':
				st.text("Number of Rows in Dataset")
				st.write(len(df_train))
			if data_dim == 'Columns':
				st.text("Number of Columns in Dataset")
				st.write(df_train.shape[1])

		if label:
			st.subheader("Label (Sentiment)")
			st.markdown('**Sentiment Classes**')
			# Number of Tweets Per Sentiment Class
			fig, axis = plt.subplots(ncols=2, figsize=(10, 5))
			ax = sns.countplot(x='sentiment',data=df_train,palette='winter',ax=axis[0])
			axis[0].set_title('Number of Tweets Per Sentiment Class',fontsize=14)
			axis[0].set_xlabel('Sentiment Class')
			axis[0].set_ylabel('Tweets')
			for p in ax.patches:
				ax.text(p.get_x() + p.get_width()/2., p.get_height(), '%d' % int(p.get_height()), fontsize=11, ha='center', va='bottom')
			raw['sentiment'].value_counts().plot.pie(autopct='%1.1f%%',colormap='winter_r',ax=axis[1])
			axis[1].set_title('Proportion of Tweets Per Sentiment Class',fontsize=14)
			axis[1].set_ylabel('Sentiment Class')
			st.pyplot()
			st.markdown('**Vader Sentiment Analysis**')
			vader_tweets = Image.open('resources/imgs/vader_tweets.png')
			st.image(vader_tweets,use_column_width=True)
			vader_tweets_prop = Image.open('resources/imgs/vader_tweets_prop.png')
			st.image(vader_tweets_prop,use_column_width=True)
			vader_tweets_pie = Image.open('resources/imgs/vader_tweets_pie.png')
			st.image(vader_tweets_pie,use_column_width=True)
			st.markdown('**Subjectivity**')
			subjectivity = Image.open('resources/imgs/subjectivity.png')
			subjectivity = subjectivity.resize((500,300))
			st.image(subjectivity)

		if wordclouds:
			st.subheader("Wordclouds")
			st.markdown('**Wordcloud for Training Dataset**')
			train_wc = Image.open('resources/imgs/train_wc.png')
			st.image(train_wc,use_column_width=True)
			st.markdown('**Wordcloud Per Sentiment Class**')
			all_wc = Image.open('resources/imgs/all_wc.png')
			st.image(all_wc,use_column_width=True)
			# df = df_train.copy()
			# df['noise'] = df['message'].apply(remove_noise)
			# train_words = wc(df['noise'])
			# train_wordcloud = WordCloud(width=1500, height=700, background_color='white', colormap='winter', min_font_size=10).generate(train_words)
			# plt.figure(figsize = (15, 7), facecolor = None)
			# plt.title("Full Dataset",fontsize=20)
			# plt.imshow(train_wordcloud)
			# plt.axis("off")
			# plt.tight_layout(pad = 0)
			# st.pyplot()
			#
			# fig, axis = plt.subplots(nrows=2, ncols=2, figsize=(18, 12))
			# news = wc(df['noise'][df['sentiment']==2])
			# news_wordcloud = WordCloud(width=900, height=600, background_color='white', colormap='winter').generate(news)
			# axis[0, 0].imshow(news_wordcloud)
			# axis[0, 0].set_title('News (2)',fontsize=14)
			# axis[0, 0].axis("off")
			# neutral = wc(df['noise'][df['sentiment']==0])
			# neutral_wordcloud = WordCloud(width=900, height=600, background_color='white', colormap='winter', min_font_size=10).generate(neutral)
			# axis[1, 0].imshow(neutral_wordcloud)
			# axis[1, 0].set_title('Neutral (0)',fontsize=14)
			# axis[1, 0].axis("off")
			# pro = wc(df['noise'][df['sentiment']==1])
			# pro_wordcloud = WordCloud(width=900, height=600, background_color='white', colormap='winter', min_font_size=10).generate(pro)
			# axis[0, 1].imshow(pro_wordcloud)
			# axis[0, 1].set_title('Pro (1)',fontsize=14)
			# axis[0, 1].axis("off")
			# anti = wc(df['noise'][df['sentiment']==-1])
			# anti_wordcloud = WordCloud(width=900, height=600, background_color='white', colormap='winter', min_font_size=10).generate(anti)
			# axis[1, 1].imshow(anti_wordcloud)
			# axis[1, 1].set_title('Anti (-1)',fontsize=14)
			# axis[1, 1].axis("off")
			# st.pyplot()

		if word_freq:
			st.subheader("Word Frequencies")
			word_freq = Image.open('resources/imgs/word_freq.png')
			word_freq = word_freq.resize((500,500))
			st.image(word_freq)

		if urls:
			st.subheader("URL Domains")
			urls = Image.open('resources/imgs/urls.png')
			urls = urls.resize((500,300))
			st.image(urls)

		if hashtags:
			st.subheader("Hashtags")
			st.markdown('**Top 5 Hashtags**')
			top_ht = pd.read_csv('resources/top5_ht.csv',index_col=0)
			st.write(top_ht)
			neg_ht = Image.open('resources/imgs/neg_ht.png')
			st.image(neg_ht,use_column_width=True)
			neutral_ht = Image.open('resources/imgs/neutral_ht.png')
			st.image(neutral_ht,use_column_width=True)
			pos_ht = Image.open('resources/imgs/pos_ht.png')
			st.image(pos_ht,use_column_width=True)
			news_ht = Image.open('resources/imgs/news_ht.png')
			st.image(news_ht,use_column_width=True)
			htags = Image.open('resources/imgs/hashtags.png')
			st.image(htags,use_column_width=True)

		if mentions:
			st.subheader("Mentions")
			st.markdown('**Top 5 Mentions**')
			top_ment = pd.read_csv('resources/top5_ment.csv',index_col=0)
			st.write(top_ment)
			ment = Image.open('resources/imgs/mentions.png')
			st.image(ment,use_column_width=True)

		if retweets:
			st.subheader("Retweets")
			rt = Image.open('resources/imgs/retweets.png')
			rt = rt.resize((500,300))
			st.image(rt)
			dropped_tweets = Image.open('resources/imgs/dropped_tweets.png')
			st.image(dropped_tweets,use_column_width=True)
			dropped_prop = Image.open('resources/imgs/dropped_prop.png')
			st.image(dropped_prop,use_column_width=True)
			dropped_pie = Image.open('resources/imgs/dropped_pie.png')
			st.image(dropped_pie,use_column_width=True)
			top_rt = Image.open('resources/imgs/top_retweets.png')
			st.image(top_rt,use_column_width=True)
			vader_retweets1 = Image.open('resources/imgs/vader_retweets1.png')
			vader_retweets1 = vader_retweets1.resize((800,160))
			st.image(vader_retweets1)
			vader_retweets2 = Image.open('resources/imgs/vader_retweets2.png')
			vader_retweets2 = vader_retweets2.resize((800,160))
			st.image(vader_retweets2)
			vader_retweets3 = Image.open('resources/imgs/vader_retweets3.png')
			vader_retweets3 = vader_retweets3.resize((800,160))
			st.image(vader_retweets3)
			dropped_vs_retweet = Image.open('resources/imgs/dropped_vs_retweet.png')
			st.image(dropped_vs_retweet,use_column_width=True)
			dropped_retweets = Image.open('resources/imgs/dropped_retweets.png')
			st.image(dropped_retweets,use_column_width=True)

	# Building out the "Tweet Sentiment Prediction" page
	if selection == "Tweet Sentiment Prediction":
		st.info("Choose a trained classification model to predict tweet sentiment and evaluate model performance.")
		st.sidebar.subheader("Classification model:")
		all_ml_models = ['Support Vector Classifier','Linear SVC','Logistic Regression','Multinomial Naive Bayes','K Neighbours Classifier']
		model_choice = st.sidebar.selectbox('Choose a model',all_ml_models)
		st.sidebar.subheader("Performance metrics:")
		#p_metrics = st.sidebar.multiselect('',('Classification Report','Confusion Matrix','ROC Curve & AUC'))
		report = st.sidebar.checkbox('Classification Report')
		con_mat = st.sidebar.checkbox('Confusion Matrix')
		roc_auc = st.sidebar.checkbox('ROC Curve & AUC')
		performance_metrics = ['Classification Report','Confusion Matrix','ROC Curve & AUC']
		# Creating a text box for user input
		tweet_text = st.text_area("Enter Text Below","")
		final = clean(tweet_text)
		tweet_text = lemmatise_words(final)

		# Load your .pkl file with the model of your choice + make predictions
		# Try loading in multiple models to give the user a choice
				# Compare Weighted F1-Scores Between Models
		if not st.button("Classify"):
			fig,axis = plt.subplots(figsize=(10, 5))
			rmse_x = ['SVC','Linear SVC','Logistic Regression','Multinomial NB','K Neighbours']
			rmse_y = [0.78,0.75,0.75,0.68,0.46]
			ax = sns.barplot(x=rmse_x, y=rmse_y,palette='winter')
			plt.title('Weighted F1-Score Per Classification Model',fontsize=14)
			plt.ylabel('Weighted F1-Score')
			for p in ax.patches:
				ax.text(p.get_x() + p.get_width()/2, p.get_y() + p.get_height(), round(p.get_height(),2), fontsize=12, ha="center", va='bottom')
			st.pyplot()
		else:
			# Transforming user input with vectorizer
			vect_text = tweet_cv.transform([tweet_text]).toarray()
			if model_choice == 'Linear SVC':
				predictor = joblib.load(open(os.path.join("resources/models/linsvc_model.pkl"),"rb"))
				prediction = predictor.predict(vect_text)
				con_mat_path = 'resources/imgs/linsvc_con_mat.png'
				report_path = 'resources/reports/linsvc_report.csv'
				roc_auc_path = 'resources/imgs/linsvc_roc_auc.png'
			elif model_choice == 'Logistic Regression':
				predictor = joblib.load(open(os.path.join("resources/models/logreg_model.pkl"),"rb"))
				prediction = predictor.predict(vect_text)
				con_mat_path = 'resources/imgs/logreg_con_mat.png'
				report_path = 'resources/reports/logreg_report.csv'
				roc_auc_path = 'resources/imgs/logreg_roc_auc.png'
			elif model_choice == 'Multinomial Naive Bayes':
				predictor = joblib.load(open(os.path.join("resources/models/multinb_model.pkl"),"rb"))
				prediction = predictor.predict(vect_text)
				con_mat_path = 'resources/imgs/multinb_con_mat.png'
				report_path = 'resources/reports/multinb_report.csv'
				roc_auc_path = 'resources/imgs/multinb_roc_auc.png'
			elif model_choice == 'Support Vector Classifier':
				predictor = joblib.load(open(os.path.join("resources/models/svc_model.pkl"),"rb"))
				prediction = predictor.predict(vect_text)
				con_mat_path = 'resources/imgs/svc_con_mat.png'
				report_path = 'resources/reports/svc_report.csv'
				roc_auc_path = 'resources/imgs/svc_roc_auc.png'
			elif model_choice == 'K Neighbours Classifier':
				predictor = joblib.load(open(os.path.join("resources/models/kn_model.pkl"),"rb"))
				prediction = predictor.predict(vect_text)
				con_mat_path = 'resources/imgs/kn_con_mat.png'
				report_path = 'resources/reports/kn_report.csv'
				roc_auc_path = 'resources/imgs/kn_roc_auc.png'

			for key,value in labels.items():
				if prediction == value:
					result = key
					st.success("Predicted Tweet Sentiment Class: {}".format(result))
			if con_mat:
				st.markdown('**Confusion Matrix**')
				con_mat_show = Image.open(con_mat_path)
				st.image(con_mat_show)#,use_column_width=True)
			if report:
				st.markdown('**Classification Report**')
				report_show = pd.read_csv(report_path,index_col=0)
				st.write(report_show)

			if roc_auc:
				st.markdown('**ROC Curve & AUC**')
				roc_auc_show = Image.open(roc_auc_path)
				st.image(roc_auc_show)
		# st.markdown("## Party time!")
		# st.write("Yay! You're done with this tutorial of Streamlit. Click below to celebrate.")
		# btn = st.button("Celebrate!")
		# if btn:
		# 	st.balloons()
# Required to let Streamlit instantiate our web app.
if __name__ == '__main__':
	main()
