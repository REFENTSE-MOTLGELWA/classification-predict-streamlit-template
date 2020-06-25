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


	# Creating predictions side bar
	model_page = ["Home", "Lsvc", "Knn"]
	st.sidebar.title("PREDICTIONS")
	selection = st.sidebar.radio("Choose Model", model_page)

	if selection == 'Home':
		st.info("Welcome to climate change sentiment analysis")
		st.info("""Please select a model under PREDICTIONS to predict a tweet 
		or make a selection under INFORMATION to get more information about the site, the exploratory data analysis and preprocessing.""")

		
	# Building out the predication page

	#Linear Support Vector classifier
	if selection == "Lsvc":
		st.info("Give brief info about Lsvc model")
		# Creating a text box for user input
		tweet_text = st.text_area("Enter Text","Type Here")
		
		if st.button("Classify"):
			# Transforming user input with vectorizer
			### vect_text = tweet_cv.transform([tweet_text]).toarray()
			# Load your .pkl file with the model of your choice + make predictions
			# Try loading in multiple models to give the user a choice
            
			path = 'resource2/Models'

			predictor = joblib.load(open(os.path.join(path,"LSVC_01.pkl"),"rb"))
			prediction = predictor.predict([tweet_text])
			if prediction == [1]:
				results = 'Pro'
			elif prediction == [-1]:
				results = 'Anti'
			elif prediction == [2]:
				results = 'News'
			elif prediction == [0]:
				results = 'Neutral'

			# When model has successfully run, will print prediction
			# You can use a dictionary or similar structure to make this output
			# more human interpretable.
			st.success("Text Categorized as:   {}".format(results))

	#K Nearest Neighbor
	elif selection == 'Knn':
		st.info("Give brief info about Knn model")
		# Creating a text box for user input
		tweet_text = st.text_area("Enter Text","Type Here")

		if st.button("Classify"):
			# Transforming user input with vectorizer
			### vect_text = tweet_cv.transform([tweet_text]).toarray()
			# Load your .pkl file with the model of your choice + make predictions
			# Try loading in multiple models to give the user a choice
            
			path = 'resource2/Models'

			predictor = joblib.load(open(os.path.join(path,"KNN_01.pkl"),"rb"))
			prediction = predictor.predict([tweet_text])

			if prediction == [1]:
				results = 'Pro'
			if prediction == [-1]:
				results = 'Anti'
			if prediction == [2]:
				results = 'News'
			if prediction == [0]:
				results = 'Neutral'

			# When model has successfully run, will print prediction
			# You can use a dictionary or similar structure to make this output
			# more human interpretable.
			st.success("Text Categorized as: {}".format(results))



	# Building out the "Information" page
	st.sidebar.title("INFORMATION")
	options = ["General", "Readme", "EDA", "Preprocessing"]
	select_info = st.sidebar.radio("Choose Option", options)
	# Building out the "Information" page
	if select_info == "General":
		st.info("General Information")
		# You can read a markdown file from supporting resources folder
		st.markdown("Some information here")

		st.subheader("Raw Twitter data and label")
		if st.checkbox('Show raw data'): # data is hidden if box is unchecked
			st.write(raw[['sentiment', 'message']]) # will write the df to the page


	if select_info == "Readme":
		st.info("README")
		# You can read a markdown file from supporting resources folder
		st.markdown(open("README.md").read())

		

	if select_info == "EDA":
		st.info("EDA")
		#our_eda = eda.explore_data()
		#st.success(our_eda)
		st.markdown("Display the EDA here")



	if select_info == "Preprocessing":
		st.info("Preprocessing")
		# You can read a markdown file from supporting resources folder
		st.markdown("Display the PREPROCESSING here")


		
# Required to let Streamlit instantiate our web app.  
if __name__ == '__main__':
	main()