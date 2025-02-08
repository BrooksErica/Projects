Classification of Firewall Log Files 

Introduction

Companies configure firewalls for their network to monitor and control what type communications are allowed and not allowed. Firewall devices use a set of security policies to characterize and manage any data that comes across the network. Configuration of firewalls are very important and vital for secure communications.  

In this case study, the goal is to classify firewall log files using a multiclass Support Vector Machine (SVM) and Stochastic Gradient Descent. The data will be grouped into one of the four classes: ‘allow’, ‘deny’, ‘drop’, and ‘reset-both.’

Method

The data used is from a Firewall Device used at Firat University. System administrator’s set-up a set of security policies based on the organization’s needs. The sample data consists of a collection of 65532 firewall log files examined over 12 different features which include the target feature ‘Action.’  The Action feature is the column of classes that we are trying to predict.
	

Action	Description
Allow	Allows the internet traffic.
Deny	Blocks traffic and enforces the default Deny Action defined for the application that is being denied.
Drop	Silently drops the traffic for an application, overrides the default deny action. A TCP reset is not sent to the host/application. 
Reset-Both	Sends a TCP reset to both the client-side and server-side devices.



Data Preprocessing and EDA

There were no missing values. Below is the distribution of the target classes. From the visual it appears that most of the data that came across the network were allowable internet traffic. The t-SNE diagram shows the clustering and relationship between the different classes.

<img width="266" alt="image" src="https://github.com/user-attachments/assets/4752e3bf-889e-40ec-bd56-cb07cecef301" />

  
Diagram 1: Distribution of Classes

 <img width="431" alt="image" src="https://github.com/user-attachments/assets/e1daa29e-095c-4bb1-9bd8-4ab88f7f3a91" />

Diagram 2: Dimensionality and Clustering of the different Classes

To see if there were any relationships between the 11 features a correlation map was constructed. There appears to be a strong correlation between the ‘Bytes’ and ‘Packets’ features.  

 <img width="338" alt="image" src="https://github.com/user-attachments/assets/4cc60750-c782-4209-99b2-8cf7c6453e19" />



Feature Selection

Our log file data was split into a training and test set, with 80% trained and 20% test before applying logistic regression + Elastic Net regularization. Elastic Net regularization uses a combination of L1 (Lasso) and L2 (Ridge) penalties to determine the most important features. It was determined that 5 of the 11 features were most important. 

	Most Important Feature
1	Source Port
2	Destination Port
3	NAT Source Port
4	NAT Destination Port
5	Elapsed Time
 	 



Model – Building

A Multiclass Support Vector Machine was used to classify the four classes. Four SVM models we built each using one of the activation functions (Linear, RBF, Poly, and Sigmoid). Split train/test data set was used to train these models 80% train and 20% test. 

We also used Stochastic Gradient Descent (SGD) for classification of the classes. The model was trained over all the data points with a loss function ‘hinge’, cross validation of 0.1, and over a total of 5 epochs, with early stopping enabled. The performance of all the models were evaluated using accuracy, recall, precision and the F1 score. ROC curves were produced to graphical show the performance of each model.

Results

The SVM Linear and SVM RBF classifiers both performed about the same. They both achieved an accuracy of 99%, and their recall, precision, and F1 scores were also the same, all at 74%. According to the confusion matrices for both models about 12,968 log files were accurately classified or grouped in their correct classes.  Whereas, about 140 were misclassified. SVM Sigmoid performed the worse out of all the models. 

*See Confusion Matrices and AUC-ROC Curves for each model below.

Method	Accuracy	Recall	Precision	F1 Score
SVM Linear	99%	74%	74%	74%
SVM RBF	99%	74%	74%	74%
SVM Polynomial	98%	74%	73%	73%
SVM Sigmoid	83%	61%	61%	61%
SGD	98%	73%	73%	73%
















SVM Linear Confusion Matrix and AUC-ROC Curve

Confusion Matrix:
 [[7521   21    3    0]
 [   0 2885  109    0]
 [   0    0 2562    0]
 [   3    3    0    0]]

<img width="199" alt="image" src="https://github.com/user-attachments/assets/11112968-5738-44f2-96e1-eb35f07c229f" />  <img width="216" alt="image" src="https://github.com/user-attachments/assets/628e855b-84c4-4516-aad0-a6c37e0f5463" />


          
SVM RBF Confusion Matrix and AUC-ROC Curve

Confusion Matrix:
 [[7512   30    3    0]
 [   0 2898   96    0]
 [   0    0 2562    0]
 [   3    3    0    0]]

         

SVM Polynomial Confusion Matrix and AUC-ROC Curve

Confusion Matrix:
 [[7426   41   78    0]
 [   0 2878  116    0]
 [   0    0 2562    0]
 [   3    3    0    0]]

       


SVM Sigmoid Confusion Matrix and AUC-ROC Curve

Confusion Matrix:
 [[6586  558  401    0]
 [ 403 2308  283    0]
 [ 525    0 2037    0]
 [   3    3    0    0]]

       
SGD Confusion Matrix and AUC-ROC Curve

Confusion Matrix:
 [[7464   19   62    0]
 [  36 2789  169    0]
 [   0    0 2562    0]
 [   3    3    0    0]]
