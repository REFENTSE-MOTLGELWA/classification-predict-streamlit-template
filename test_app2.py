'''
This is a version of base_app that in a pipeline that vectorizes 
and fits the data to the model the pipeline wwas build with.
'''

# Streamlit dependencies
import streamlit as st
import joblib,os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import nltk
#import spacy
import re
#import emoji

# For printing option and text color
class color:
    PURPLE = '\033[95m'
    CYAN = '\033[96m'
    DARKCYAN = '\033[36m'
    BLUE = '\033[94m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'
    END = '\033[0m'



raw = pd.read_csv('resource2/datasets/train.csv')

#df_sub = pd.read_csv('resource2/datasets/sample_submission.csv')
df_test = pd.read_csv('resource2/datasets/test.csv')
test_df = df_test.set_index('tweetid')
df_train = pd.read_csv('resource2/datasets/train.csv')
train_df = df_train.set_index('tweetid')


head_train = train_df.head()
head_train_shape = train_df.shape
head_test = test_df.head()
head_test_shape = test_df.shape

 

missing_train = train_df.isna().sum()
missing_test = test_df.isna().sum()

# Count of classes in sentiment 
sns.set(style="darkgrid",palette='summer')
ax = sns.countplot(x='sentiment', data=train_df)
	

def empty_message():
	blanks_test = []
	for tID,msg in test_df.itertuples():
		if msg.isspace == True:
			blanks_test.append(tID)

	blanks_train = []
	for tID,sent,msg in train_df.itertuples():
		if msg.isspace == True:
			blanks_test.append(tID)
	return blanks_train, blanks_test

def null_value_check():
	print(f'No. of empty messages in train: {len(empty_message.blanks_train)}\n')
	print(f'No. of empty messages in test: {len(empty_message.blanks_test)}')




def class_dist_perct():
	print(color.BOLD +'Percentage of a particular `Class` in the train dataset\n'+ color.END)
	print(f'Class 2 ~ News \n{round((df_train.sentiment.value_counts()[2]/len(df_train))*100,2)} %\n')
	print(f'Class 1 ~ Pro \n{round((df_train.sentiment.value_counts()[1]/len(df_train))*100,2)} %\n')
	print(f'Class 0 ~ Neutral \n{round((df_train.sentiment.value_counts()[0]/len(df_train))*100,2)} %\n')
	print(f'Class -1 ~ Anti \n{round((df_train.sentiment.value_counts()[-1]/len(df_train))*100,2)} %')


# The main function where we will build the actual app
def main():
	"""Tweet Classifier App with Streamlit """

	# Creates a main title and subheader on your page -
	# these are static across all pages
	img = "resource2/images/logo.jpg"
	st.image(img, caption="It is what it is")
	st.title("Tweet Classifer")
	#st.subheader("Climate change tweet classification")


	# Creating side bar
	navigation = ["HOME", "PREDICTIONS", "INSIGHTS", "EDA", "ABOUT TEAM", "READMEfile"]
	st.sidebar.title("NAVIGATION")
	selection = st.sidebar.radio("make your selection",navigation)

	# For selecting side bars
	if selection == 'HOME':
		st.markdown(open("resource2/info.md").read())
	

		
	# Building out the predication page
	elif selection == "PREDICTIONS":
		#Creating a text box for user input
		tweet_text = st.text_area("Enter Text","Type Here")
		#List of models
		model_list = ["LSVC", "KNN"]
		#Model selection
		model_select = st.selectbox("Choose Model", model_list)

		#Linear Support Vector classifier
		if model_select == "LSVC":
			#What is the model about
			st.info("Give brief info about Linear Support Vector Classifier model")
			#
			if st.button("Classify"):
				# Transforming user input with vectorizer
				### vect_text = tweet_cv.transform([tweet_text]).toarray()
				# Load your .pkl file with the model of your choice + make predictions
				# Try loading in multiple models to give the user a choice
				
				path = 'resource2/Models'

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
			st.info("Give brief info about K Nearest Neighbor model")

			if st.button("Classify"):
				# Transforming user input with vectorizer
				### vect_text = tweet_cv.transform([tweet_text]).toarray()
				# Load your .pkl file with the model of your choice + make predictions
				# Try loading in multiple models to give the user a choice
				
				path = 'resource2/Models'

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
		st.subheader("Meaning of different output")
		st.markdown("Pro - the tweet supports the belief of man-made climate change")
		st.markdown("Anti - the tweet does not believe in man-made climate change")
		st.markdown("Neutral - the tweet neither supports nor refutes the belief of man-made climate change")
		st.markdown("News - the tweet links to factual news about climate change")

	
	# Building out the "Information" page
	elif selection == "INSIGHTS":
		st.info("Our insights goes here")

	elif selection == "EDA":
		st.markdown("Display the EDA here")
		st.subheader("Raw Twitter data and label")
		if st.checkbox('Show raw data'): # data is hidden if box is unchecked
			sub_data = ["Train Head Data", "Train Data Shape", "Test Head Data", "Test Data Shape"]
			st.write(raw[['sentiment', 'message']]) # will write the df to the page
			data_select = st.selectbox("Choose data", sub_data)
			if data_select == "Train Head Data":
				st.write(head_train)

			elif data_select == "Train Data Shape":	
				st.write(head_train_shape)

			elif data_select == "Test Head Data":
				st.write(head_test)

			elif data_select == "Test Data Shape":
				st.write(head_test_shape)

		
		elif st.checkbox('Display sentiment distribution'): # data is hidden if box is unchecked
			st.write(ax)

	elif selection == "ABOUT TEAM":
		#Display info about team 
		st.markdown(open("team.md").read())

	elif selection == "READMEfile":
		#Display readme page
		st.markdown(open("README.md").read())

		
# Required to let Streamlit instantiate our web app.  
if __name__ == '__main__':
	main()