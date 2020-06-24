'''
This is a version of base_app that in a pipeline that vectorizes 
and fits the data to the model the pipeline wwas build with.
'''

# Streamlit dependencies
import streamlit as st
import joblib,os

import resource2.pages.home
import resource2.pages.knn
import resource2.pages.lsvc
import eda


# Data dependencies
import pandas as pd


raw = pd.read_csv('Kaggle_resources/datasets/train.csv')

model_page = {
    "Home": resource2.pages.home,
    "KNN": resource2.pages.knn,
    "LSVC": resource2.pages.lsvc,
}

# The main function where we will build the actual app
def main():
	"""Tweet Classifier App with Streamlit """

	# Creates a main title and subheader on your page -
	# these are static across all pages
	st.title("Tweet Classifer")
	st.subheader("Climate change tweet classification")

	# Creating sidebar with selection box -
	# you can create multiple pages this way
	st.sidebar.title("PREDICTIONS")
	selection = st.sidebar.selectbox("Choose Model", list(model_page.keys()))
	
	page = model_page[selection]

	# Building out the predication page
	if selection == 'KNN':
		st.info("Prediction with KNN Model")
		st.write(page)
		# Creating a text box for user input
		tweet_text = st.text_area("Enter Text","Type Here")

		if st.button("Classify"):
			# Transforming user input with vectorizer
			### vect_text = tweet_cv.transform([tweet_text]).toarray()
			# Load your .pkl file with the model of your choice + make predictions
			# Try loading in multiple models to give the user a choice
            
			path = 'Kaggle_resources/Models'

			predictor = joblib.load(open(os.path.join(path,"KNN_01.pkl"),"rb"))
			prediction = predictor.predict([tweet_text])

			# When model has successfully run, will print prediction
			# You can use a dictionary or similar structure to make this output
			# more human interpretable.
			st.success("Text Categorized as: {}".format(prediction))


	elif selection == "LSVC":
		st.info("Prediction with LSVC Model")
		st.write(page)
		# Creating a text box for user input
		tweet_text = st.text_area("Enter Text","Type Here")

		if st.button("Classify"):
			# Transforming user input with vectorizer
			### vect_text = tweet_cv.transform([tweet_text]).toarray()
			# Load your .pkl file with the model of your choice + make predictions
			# Try loading in multiple models to give the user a choice
            
			path = 'Kaggle_resources/Models'

			predictor = joblib.load(open(os.path.join(path,"LSVC_01.pkl"),"rb"))
			prediction = predictor.predict([tweet_text])

			# When model has successfully run, will print prediction
			# You can use a dictionary or similar structure to make this output
			# more human interpretable.
			st.success("Text Categorized as: {}".format(prediction))

	# Building out the "Information" page
	st.sidebar.title("INFORMATION")
	options = ["General", "Readme", "EDA", "Preprocessing"]
	select_info = st.sidebar.selectbox("Choose Option", options)
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
		our_eda = eda.explore_data()
		st.success(our_eda)



	if select_info == "Preprocessing":
		st.info("Preprocessing")
		# You can read a markdown file from supporting resources folder
		st.markdown("Put the PREPROCESSING here")


		
# Required to let Streamlit instantiate our web app.  
if __name__ == '__main__':
	main()