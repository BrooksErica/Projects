Spam or No Spam Email Filtering Classifier Model

Introduction

It is not unusual to receive hundreds of emails a day. It can be time consuming and difficult trying to read each email and determine if it is spam or a real email. By training a model to filter through and classify the different emails would make this process much easier. In this case study, the goal is to use a classifier like Multinomial Naïve Bayes to filter through email text and detect whether the email is spam or ham (real email).

Method

The data used is from the Apache SpamAssassin open-source project. System administrators use a filtering system to classify wanted email and unsolicited email. The sample data consists of a collection of email text contained in five folders. Three of the folders are ‘ham’ (not spam or wanted emails) and two of the folders are ‘spam.’ 

| Directory/Folders |	Description |
| ----------------- | ----------- |
| hard_ham |	Not spam |
| easy_ham |	Not spam |
| easy_ham_2 |	Not spam |
| spam |	Spam |
| Spam_2 |	Spam |

Functions

Two functions were created, one to handle loading the emails for analysis and the second for preprocessing the email text. 

1.	Load_Emails(directory)
-	The function reads through all the email files in each directory, extracts the body content, and labels the emails based on whether the directory name contains 'spam' or 'ham'. It returns a list of tuples where each tuple contains the email body and its corresponding label (1 for spam, 0 for ham).

2.	Preprocess_text(text)
-	This function is designed to clean and prepare the data/email text for analysis. It uses natural language processing (NLP) tasks to remove HTML tags, puntuations, and stop words. It also reduces words to their root form.



The emails were loaded and preprocessed for an overall total of 9353 email text. 2399 (25.6%) are labeled as spam and 6954 (74.4%) are labeled as not spam.  

<img width="242" alt="image" src="https://github.com/user-attachments/assets/f1dc57aa-81ca-4d80-ac94-48f923268004" />



CountVectorizer was used as a feature extraction. The email text data was transformed into a matrix of counts (‘bag-of-words”) or numerical representation to be used for modeling.  

To visualize the features and see the clustering or shape of data a DBSCAN was performed. In the clustering graph below we can see that the emails labeled as spam are classified by the brown color and the emails labeled as not-spam are classified by the blue color.


<img width="282" alt="image" src="https://github.com/user-attachments/assets/79e64598-fe92-4a40-b6b6-d80c274ce436" />


 

Model – Building

Our email/text data was split into a training and test set, with 80% trained and 20% test. 
Multinomial Naïve Bayes classifier was used to build and train our model. The model was evaluated using accuracy.

Results

The Naïve Bayes classifier achieved an accuracy of 98%. It was able to correctly classify 98 out of 100 emails. According to the confusion matrix 1381 emails was accurately identified as not-spam (ham) while 4 was incorrectly identified as spam. Similarly, 453 were accurately identified as spam while 33 were incorrectly identified as not-spam (ham). 

Accuracy: 0.980224478888295
Classification Report:
              precision    recall  f1-score   support

           0       0.98      1.00      0.99      1385
           1       0.99      0.93      0.96       486

    accuracy                           0.98      1871
   macro avg       0.98      0.96      0.97      1871
weighted avg       0.98      0.98      0.98      1871


Confusion Matrix:
 [[ 453   33]
 [   4 1381]]

<img width="292" alt="image" src="https://github.com/user-attachments/assets/0103a891-69af-44d5-8a0c-6210fee00f15" />



