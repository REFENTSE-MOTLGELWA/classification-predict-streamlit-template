'''
This is a version of base_app that in a pipeline that vectorizes 
and fits the data to the model the pipeline wwas build with.
'''

# Streamlit dependencies
import streamlit as st
import joblib,os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import string
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import TreebankWordTokenizer
from nltk import SnowballStemmer
from nltk.stem import WordNetLemmatizer
#import spacy
#from PIL import Image

#import dataset
df_test = pd.read_csv('resources/datasets/test.csv')
test_df = df_test.set_index('tweetid')
df_train = pd.read_csv('resources/datasets/train.csv')
train_df = df_train.set_index('tweetid')
 



# The main function where we will build the actual app
def main():
	"""Tweet Classifier App with Streamlit """

	# Creates a main title and subheader on your page -
	# these are static across all pages
	img = "resources/images/logo.jpg"
	st.image(img, caption="It is what it is")
	st.title("Tweet Classifer")

	# Creating side bar for navigation
	navigation = ["HOME", "PREDICTIONS", "EDA", "INSIGHTS", "ABOUT TEAM", "READMEfile"]
	st.sidebar.title("NAVIGATION")
	selection = st.sidebar.radio(" ",navigation)


	# For selecting side bars
	if selection == 'HOME':
		st.markdown(open("resources/info.md").read())
		st.subheader("Raw Twitter data and label")
		
		if st.checkbox('Show raw data'): # data is hidden if box is unchecked
			values = st.slider("Slide to the right to select and view quantity",0, 16000)
			st.write(df_train[0:values])

			sub_data = ["Train Data Shape", "Test Data Shape"]
			data_select = st.selectbox("View Data Shape", sub_data)

			if data_select == "Train Data Shape":	
				st.write(df_train.shape)

			elif data_select == "Test Data Shape":
				st.write(df_test.shape)
	

	# Building out the predication page
	elif selection == "PREDICTIONS":
		#Creating a text box for user input
		tweet_text = st.text_area("Enter some text below:","")
		#List of models
		model_list = ["LSVC", "KNN", "MultiNB"]
		#Model selection drop down
		model_select = st.selectbox("Choose Model:", model_list)

		#Linear Support Vector classifier
		if model_select == "LSVC":
			#What is the model about
			st.markdown("replace text with description of Linear Support Vector Classifier model")
			#
			if st.button("Classify"):
				# Transforming user input with vectorizer
				### vect_text = tweet_cv.transform([tweet_text]).toarray()
				# Load your .pkl file with the model of your choice + make predictions
				# Try loading in multiple models to give the user a choice
				
				path = 'resources/models'

				predictor = joblib.load(open(os.path.join(path,"LSVC_01.pkl"),"rb"))
				#change prediction out to display in more human interpretable
				prediction = predictor.predict([tweet_text])
				if prediction == [1]:
					results = 'Pro'
				elif prediction == [-1]:
					results = 'Anti'
				elif prediction == [2]:
					results = 'News'
				elif prediction == [0]:
					results = 'Neutral'
				#prediction output
				st.success("Text Categorized as:   {}".format(results))

		#K Nearest Neighbor
		elif model_select == 'KNN':
			st.markdown("replace text with description of K Nearest Neighbor model")

			if st.button("Classify"):
				# Transforming user input with vectorizer
				### vect_text = tweet_cv.transform([tweet_text]).toarray()
				# Load your .pkl file with the model of your choice + make predictions
				# Try loading in multiple models to give the user a choice
				
				path = 'resources/models'

				predictor = joblib.load(open(os.path.join(path,"KNN_01.pkl"),"rb"))
				prediction = predictor.predict([tweet_text])
				#change prediction out to display in more human interpretable 
				if prediction == [1]:
					results = 'Pro'
				if prediction == [-1]:
					results = 'Anti'
				if prediction == [2]:
					results = 'News'
				if prediction == [0]:
					results = 'Neutral'
				#prediction output	
				st.success("Text Categorized as: {}".format(results))


		elif model_select == 'MultiNB':
			st.markdown("replace text with description of MultiNB model")

			if st.button("Classify"):
				# Transforming user input with vectorizer
				### vect_text = tweet_cv.transform([tweet_text]).toarray()
				# Load your .pkl file with the model of your choice + make predictions
				# Try loading in multiple models to give the user a choice
				
				path = 'resources/models'

				predictor = joblib.load(open(os.path.join(path,"MultiNB.pkl"),"rb"))
				prediction = predictor.predict([tweet_text])
				#change prediction out to display in more human interpretable 
				if prediction == [1]:
					results = 'Pro'
				if prediction == [-1]:
					results = 'Anti'
				if prediction == [2]:
					results = 'News'
				if prediction == [0]:
					results = 'Neutral'
				#prediction output	
				st.success("Text Categorized as: {}".format(results))


		st.subheader("Meaning of different categories")
		st.info("Pro - the tweet supports the belief of man-made climate change")
		st.info("Anti - the tweet does not believe in man-made climate change")
		st.info("Neutral - the tweet neither supports nor refutes the belief of man-made climate change")
		st.info("News - the tweet links to factual news about climate change")

	
	# Building out the "Information" paga
		
	elif selection == "EDA":
		st.subheader("Exploratry Data Analysis")
		
		if st.checkbox('Display sentiment distribution'): # data is hidden if box is unchecked
			# Count of classes in sentiment 
			plt.figure(figsize=(10, 6))
			sns.set(style="darkgrid")
			ax = sns.countplot(x='sentiment', data=train_df)
			ax.set_title("No. of tweets per sentiment")
			st.pyplot()

		
			df_train['length'] = df_train['message'].apply(len)
			graph = sns.FacetGrid(data=df_train, col = 'sentiment', 
								col_wrap=2, height=3, aspect=2)
			graph.map(plt.hist, 'length', bins = 30, color = 'g')
			st.pyplot()




	elif selection == "INSIGHTS":
		st.info("Our insights goes here")
		news_class = round((train_df.sentiment.value_counts()[2]/len(train_df))*100,2)
		pro_class = round((train_df.sentiment.value_counts()[1]/len(train_df))*100,2)
		neutral_class = round((train_df.sentiment.value_counts()[0]/len(train_df))*100,2)
		anti_class = round((train_df.sentiment.value_counts()[-1]/len(train_df))*100,2)

		data = [news_class,pro_class, neutral_class, anti_class]
		classes = 'News','Pro','Neutral','Anti'
		my_colors = ['lightblue','lightyellow','pink', 'violet']
		plt.pie(data,labels=classes,autopct='%1.1f%%', colors=my_colors)
		plt.title('Sentiment distribution in train data')
		plt.axis('equal')
		st.pyplot()

	elif selection == "ABOUT TEAM":
		#Display info about team 
		st.markdown(open("team.md").read())

	elif selection == "README":
		#Display readme page
		st.markdown(open("README.md").read())

		
# Required to let Streamlit instantiate our web app.  
if __name__ == '__main__':
	main()