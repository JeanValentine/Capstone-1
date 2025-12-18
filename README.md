# Overview
The objective of this capstone project is to enable learners to apply Natural Language Processing (NLP), Machine Learning (ML), and Deep Learning (DL) techniques to analyze customer reviews, predict product ratings, detect fake/spam reviews, and generate product insights from structured and unstructured data.

Learners will work on an end-to-end AI pipeline, including data cleaning, feature engineering, model development, and deployment, preparing them for real-world applications in e-commerce, marketing analytics, and product sentiment analysis.

## Project Statement

E-commerce platforms host thousands of customer reviews that influence purchase decisions. However, extracting meaningful insights from these reviews poses significant challenges:

  * Understanding customer sentiment across various product categories.
  * Predicting product ratings from review text and structured data.
  * Detecting fake/spam reviews based on unnatural patterns in user feedback.
  * Summarizing customer reviews to provide concise insights for buyers.

This capstone will focus on ML, NLP, and deep learning techniques to address these challenges.

## Data Description:

### Data Description

The dataset contains structured and unstructured data, including:

### Structured Data

  * id: Unique identifier for the review.
  * brand: Brand of the product.
  * categories: Product categories (e.g., Electronics, Food, Music).
  * manufacturer: Manufacturer of the product.
  * manufacturerNumber: Manufacturer’s product code.
  * reviews.numHelpful: Number of helpful votes received.
  * reviews.rating: Star rating (1-5 scale).
  * reviews.date: Date when the review was written.
  * reviews.didPurchase: Whether the customer purchased the product.
  * reviews.doRecommend: Whether the reviewer recommends the product.

### Unstructured Data

  * reviews.text: Full-text customer review.
  * reviews.title: Short title of the review.
  * reviews.username: Name of the reviewer.
  * reviews.userCity & reviews.userProvince: User’s location.
  * reviews.sourceURLs: Source link to the review.

This dataset presents opportunities for predictive modeling, NLP, sentiment analysis, and fake review detection.

### Target Variable for Fake Review Detection

Since the dataset does not explicitly contain a fake review label, we will derive a proxy target variable using one or more of the following methods:

1. Using "reviews.numHelpful" as an Indicator
* Assumption: Fake reviews often receive zero or very few helpful votes.
* Labeling Strategy:
  * Fake Review (1): reviews.numHelpful == 0
  * Genuine Review (0): reviews.numHelpful > threshold (e.g., 5 helpful votes
2. Using Review Length & Sentiment for Anomaly Detection
* Assumption: Fake reviews are often very short, overly positive/negative, or repetitive.
* Labeling Strategy:
  * Fake Review (1): Reviews with less than 5 words in reviews.text.
  * Fake Review (1): Reviews with high sentiment polarity but low helpful votes.
  * Genuine Review (0): Longer, detailed, and diverse language reviews.
3. Using Metadata Anomalies
* Assumption: Fake reviews often have suspicious metadata patterns, such as:
  * Bulk reviews posted on the same date (reviews.date).
  * Reviews from duplicate usernames (reviews.username).
  * Reviews with repetitive text across multiple products.
* Labeling Strategy: Identify and flag such reviews as potential fake reviews (1).

# Steps to Perform

## Section 1: Data Preprocessing & EDA

1. Data Preprocessing with Pandas & NumPy
* Load and explore the dataset.
* Handle missing values in reviews.text, reviews.rating, and reviews.didPurchase.
* Convert reviews.date into datetime format and analyze trends over time.
* Clean the reviews.text field by:
  * Removing special characters, stopwords, and HTML tags.
  * Tokenizing and lemmatizing the text.
2. Exploratory Data Analysis (EDA)
* Compute summary statistics for ratings, helpful votes, and purchase behavior.
* Identify the most and least reviewed brands using aggregation.
* Visualize the distribution of ratings across product categories.
* Analyze customer sentiment trends over time.
 

## Section 2: Feature Engineering & Predictive Modeling

1. Feature Engineering
* Create new features:
  * Review length (number of words in reviews.text).
  * Sentiment polarity using VADER or TextBlob.
  * Helpfulness score (ratio of helpful votes to total votes).
* Encode categorical variables like brand, categories, and reviews.doRecommend.
* Use TF-IDF or word embeddings (Word2Vec, GloVe) to convert reviews.text into numerical form.
2. Fake Review Detection with Machine Learning
* Train Logistic Regression, Decision Trees, and Random Forests to detect fake reviews.
* Use Precision, Recall, and F1-score to evaluate model performance.
* Implement Anomaly Detection models (Isolation Forest, One-Class SVM).
 

## Section 3: Deep Learning & NLP for Review Analysis

1. Detecting Fake Reviews
* Build a RNN to detect fake reveiws
* Train a BERT or Transformer-based model for fake review classification.
 

## Section 4: Building a Review Summarization App using LLMs

* Develop a Gradio-based web app where users can input multiple reviews separated by a pipe (|).
* The app should generate a summary of all reviews using an LLM or NLP model.
