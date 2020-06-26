'''
This is a version of base_app that in a pipeline that vectorizes 
and fits the data to the model the pipeline wwas build with.
'''

# Streamlit dependencies
import streamlit as st
import joblib,os


import eda


# Data dependencies
import pandas as pd


raw = pd.read_csv('resource2/datasets/train.csv')


# The main function where we will build the actual app
def main():
	"""Tweet Classifier App with Streamlit """

	# Creates a main title and subheader on your page -
	# these are static across all pages
	img = "resource2/images/logo.jpg"
	st.image(img)
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
			st.write(raw[['sentiment', 'message']]) # will write the df to the page

	elif selection == "ABOUT TEAM":
		#Display info about team 
		st.markdown("Display team info here")

	elif selection == "READMEfile":
		#Display readme page
		st.markdown(open("README.md").read())

		
# Required to let Streamlit instantiate our web app.  
if __name__ == '__main__':
	main()