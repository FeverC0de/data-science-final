# [cite_start]EVALUATING INDIVIDUAL AND SOCIOECONOMIC FACTORS IN STUDENT PERFORMANCE USING MACHINE LEARNING [cite: 748]

## INTRODUCTION
For this exploration, my partner and I decided to explore factors that influence student performance. [cite_start]We look at not only socioeconomic factors, but the individual factors which influence their grades[cite: 749, 750]. [cite_start]For this, we found a data set on Kaggle, which consists of student data showing their exam scores, as well as various other factors, such as internet access and attendance, which potentially could impact it[cite: 751].

### Exploration Question
[cite_start]In this exploration, we will be looking at which factors are important in predicting student exam scores using machine learning[cite: 752, 753].

[cite_start]The process of answering this question will involve representing the categorical data in the dataset as numerical values, using ordinal and one-hot encoding in order to prepare it to be trained on by an XGBoost machine learning model with the final exam score as the output[cite: 754]. [cite_start]The model will allow us to extract the factors which it places emphasis on when it tries to predict exam scores[cite: 755].

## METHOD AND RESULTS
[cite_start]As mentioned above, the dataset used was from Kaggle, which can be found here: [https://www.kaggle.com/datasets/lainguyn123/student-performance-factors](https://www.kaggle.com/datasets/lainguyn123/student-performance-factors)[cite: 757].

### Exploring the Dataset
[cite_start]The dataset consists of 6607 entries, with 20 columns of different factors[cite: 759]. [cite_start]The below of the column names output is retrieved by using the `df.column` method[cite: 760]:

* 'Hours_Studied'
* 'Attendance'
* 'Parental_Involvement'
* 'Access_to_Resources'
* 'Extracurricular_Activities'
* 'Sleep_Hours'
* 'Previous_Scores'
* 'Motivation_Level'
* 'Internet_Access'
* 'Tutoring_Sessions'
* 'Family_Income'
* 'Teacher_Quality'
* 'School_Type'
* 'Peer_Influence'
* 'Physical_Activity'
* 'Learning_Disabilities'
* 'Parental_Education_Level'
* 'Distance_from_Home'
* 'Gender'
* [cite_start]'Exam_Score' [cite: 761]

[cite_start]For our analysis, we will use all the columns present in our table[cite: 765]. [cite_start]However, from the table, we can observe that each column has different categorical data[cite: 766]. [cite_start]We thus cannot directly use our machine learning model on this because the data on the table must all be numeric[cite: 767]. This is because machine learning models operate using numbers. [cite_start]It’s necessary to convert these values to numbers to make it compatible with the machine learning model[cite: 768].

## [cite_start]DATA FILTERING [cite: 769]

### Handling N/A Values
[cite_start]Before converting our values, we want to make sure that there are no parts of the dataset that will cause issues[cite: 771]. [cite_start]For this, we typically want to consider if any of the rows have any missing values[cite: 772]. [cite_start]Pandas provided an easy way to verify this by using the `.isna()` method[cite: 773].

[cite_start]There seems to be missing data in the `Teacher_Quality`, `Parental_Education_Level`, and `Distance_from_Home` columns[cite: 774]. [cite_start]The total sum of cells with missing values is 229, which is around 3.4% of our dataset[cite: 775]. [cite_start]It’s not a large amount, and it’s worth considering removing it, as missing values could affect the training of our model[cite: 776].

[cite_start]Before removing it, it is important to consider that the part of the dataset we are removing could be vital to accurately training our model[cite: 777]. [cite_start]This is because the section of data we eliminate might have unique characteristics not present in the rest of the dataset[cite: 778].

[cite_start]In order to confirm there are no unique characteristics, we compare the mean exam score of our dataset without the rows with missing values, with the mean exam score of the extracted rows with null values[cite: 779, 780]. [cite_start]For this we can create a mask using the `df.isnull()` method and use the `df.describe()` method to obtain the summary stats of both sub datasets[cite: 781, 782].

[cite_start]Looking at our mean exam score for both datasets, 67.25 vs 66.78, we can see there is a difference of around ~0.47 points[cite: 786, 787]. This is not a stark difference at all. [cite_start]This is true for the different metrics as well, such as attendance, sleep hours[cite: 788]. [cite_start]The bar chart shows that other columns are not vastly different[cite: 789]. [cite_start]We can thus go ahead and drop the rows from our data set, as it will not affect the training of our machine learning model by a noticeable amount[cite: 791].

### One Hot Encoding
[cite_start]The next step in preparing our dataset is dealing with categorical data[cite: 793]. [cite_start]Given that our machine learning model cannot tell the mathematical difference between “Male” and “Female”, and given that there is no inherent order (male is not less than female and vice versa) it is necessary to use a method known as One Hot Encoding[cite: 794]. [cite_start]One Hot Encoding allows us to represent our categorical data as numerical values[cite: 795].

[cite_start]For example, for gender, we would normally create two new columns: `Is_male` and `is_female`[cite: 796, 797]. [cite_start]If the original value is “male”, we assign 1 in the `is_male` column and 0 in the `is_female` column[cite: 797]. [cite_start]However, given that adding an extra column might increase our models dimensionality and result in multicollinearity, the second column, `is_female`, is usually dropped as `is_male = 0` implies to that computer that the student is female[cite: 798].

[cite_start]Before applying One Hot Encoding we must assess which variables need to be encoded, i.e., categories which have no inherent order[cite: 799, 800].

**One Hot Encoded Columns:**
* Internet Access (Yes/No)
* Gender (Male, Female)
* Learning Disabilities (Yes/No)
* Extracurricular Activities (Yes/No)
* [cite_start]School Type - (Private/Public) [cite: 801]

[cite_start]Pandas offers a simplified way we can one hot encode our data using the `pd.get_dummies(df, columns_list, drop_first)` method, which automatically encodes the columns in the `columns_list` list[cite: 805, 806]. [cite_start]The `drop_first` parameter asks whether or not you want to drop one of the columns to avoid multicollinearity[cite: 807].

### Ordinal Encoding
[cite_start]For the remaining columns which we did not want one hot encode, we used Ordinal Encoding[cite: 808, 809]. [cite_start]For a given column with ordered categorical data, such as distance from home which has the categories (Near, Moderate, High), we can map these to numerical values based on their order[cite: 810].

* Near: 0
* Moderate: 1
* [cite_start]Far: 2 [cite: 811, 812, 813, 814, 815, 816]

[cite_start]To accomplish this, we use the `pd.Categorical()` method after creating a list of the categories in the order they are in[cite: 818].

**Ordinal Encoded Columns:**
* Access To Resources
* Motivation Level
* Teacher Quality
* Parental Involvement
* Parental Education Level
* Peer Influence
* [cite_start]Distance from home [cite: 801]

### Final Table
[cite_start]Now that we have converted all the values in our table to numerical types through Ordinal and One Hot Encoding, our data is ready to be trained on with a machine learning model[cite: 819, 820, 821].

## [cite_start]TRAINING THE MODEL [cite: 824]
[cite_start]Now that we have successfully transformed our dataset, we can start training our model[cite: 825].

### XGBoost Model
[cite_start]This is a model that combines smaller decision trees where each tree learns to correct the errors of previous trees[cite: 827]. [cite_start]I decided to use this model because it is well optimized for preventing overfitting while being able to obtain high accuracy, outperforming linear models when using complex tabular data[cite: 827].

[cite_start]Before training our model however, we must split our data set and perform cross validation[cite: 828]. [cite_start]For our model, we need to set our output of our model to be the `Exam Score` column[cite: 829]. [cite_start]The remaining columns will be the input data for our model[cite: 830].

### Feature Scaling
[cite_start]Typically, for performing linear regression, we would want to adopt feature scaling, in order for our model not to have a bias to certain categorical variables[cite: 831, 832]. [cite_start]However, given that XGBoost is a tree based model, it is not affected by the input data having different magnitudes and thus we would not need to normalize our dataset[cite: 833].

### Splitting the Dataset
[cite_start]First we need to split the dataset into a test set and training set[cite: 835]. [cite_start]Typically we should reserve around 80% of the dataset to be used for training, and the remaining 20% to be used for testing and evaluating our trained model[cite: 836].
[cite_start]Code: `X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)`[cite: 601].

### Cross-validation
[cite_start]Before we do our final training step, we must perform cross validation[cite: 602]. [cite_start]This is because we want to make sure our machine learning model generalizes evenly across the dataset and to see if our model is sensitive to one particular part of the data set[cite: 603].

[cite_start]We split our training data into 5 slices, and train a model on 4 out of those 5 slices, with the last slice being the testing set[cite: 605]. [cite_start]In subsequent iterations, we make the other slices the test sets and continue till all slices have been used as a test set[cite: 606]. [cite_start]The result of the cross validation gives us the root mean squared error (RMSE) of predicting on the test sets[cite: 607].

We obtain 5 values:
* Fold 1: 2.18462729
* Fold 2: 2.37527418
* Fold 3: 2.51594305
* Fold 4: 2.6624608
* Fold 5: 3.18173933

[cite_start]While there seems to be an increase in the RMSE per fold, they are not vastly different giving an average RMSE of 2.58[cite: 609]. [cite_start]This confirms that our XGBoost model is capable of generalizing throughout the whole dataset and we can thus move on to training our model[cite: 625].

### Final Training
[cite_start]We can thus leave our training split as is and train the final model on the dataset[cite: 626]. [cite_start]Our training gives us a model with a RMSE of 2.29 points[cite: 627]. [cite_start]This is really good, it means that predictions for our model are on average off by 2.29 points[cite: 628].

### Model Error
[cite_start]After training, we can check our prediction accuracy for the training dataset and the test data set by creating a plot of our predicted values and comparing them to the actual values[cite: 629, 630].

* [cite_start]**Training Error:** For scores below 80 points, the points follow the prediction line nicely with a low prediction error[cite: 633]. [cite_start]However past the 80 points mark, we see points reside far under the prediction line[cite: 634]. [cite_start]This indicates that there is some negative bias our model has with high exam scorers as it tends to under predict them[cite: 635]. [cite_start]This could be due to our XGBoost model optimizing against overfitting by deprioritizing extreme values[cite: 636].
* [cite_start]**Test Error:** Majority of the points seem to lie within the prediction line, however, we can see the underprediction being noticeable past the 80 point mark, which is expected as the same behaviour is observed in our training set[cite: 640].

I was curious to see how many of the data points have more than 5 point error. [cite_start]The output of the result gives us 11 (7 of those where high scores), which is ~0.94% of our data points in the test set[cite: 662, 668, 669]. [cite_start]Our model thus is capable of predicting the majority of scores less than 80 points but fails to predict students who excel[cite: 670].

## [cite_start]MODEL INTERPRETATION [cite: 672]
[cite_start]Now that we have our fully trained model, we can start to analyze which factors the model places more emphasis on[cite: 672]. Feature importance measures how useful a feature is for prediction. [cite_start]It doesn’t necessarily show correlation[cite: 673].

### Evaluating Feature Importance
[cite_start]From the graph of feature importance, we can see that two factors out of all of them dominate our dataset[cite: 675]:
1.  **Attendance:** 42.5%
2.  **Hours_Studied:** 26.8%

[cite_start]The next 4 factors contribute under 10% each: `previous_scores`, `tutoring_sessions`, `parental_involvement`, and `access_to_resources`[cite: 675]. [cite_start]The remaining factors contribute less than 2%[cite: 676]. [cite_start]This means that studying and attending school are important when it comes to predicting exam scores according to the model[cite: 676].

## [cite_start]CONCLUSIONS [cite: 693]
[cite_start]According to the model, there is a high association of student attendance and hours studied with the student exam scores[cite: 693]. [cite_start]With this Kaggle dataset used, these two factors are the highest predictors and most important features they look at[cite: 694].

[cite_start]However, this does not indicate a correlation between those factors and exam scores in the real world[cite: 697]. [cite_start]It is merely what the model looks at when trying to predict our exam scores[cite: 698]. [cite_start]Moreover, our model breaks down when it comes to predicting high scores, and does not provide a perfect predictor of student performance[cite: 699].

## [cite_start]LIMITATIONS [cite: 701]
There are a couple limitations to consider when utilising this dataset, as well with our methodology.

* [cite_start]**Unable to Accurately Predict High Performers:** The breakdown in model accuracy past the 80% mark means that it will be difficult for the model to predict the performance of gifted students[cite: 702]. [cite_start]The high performing students form a very small part of the dataset, making it difficult for the model to learn the facts which contribute to high performance[cite: 703].
* [cite_start]**Multicolinearity Effects on Factors:** The model, in my opinion, seems to place too much weightage on attendance and too little on factors such as parental motivation and sleep hours, both of which, logically should have a strong correlation with student performance[cite: 704]. [cite_start]It was possible that sleep hours and parental motivation could be correlated to attendance creating an implicit multicollinearity where our Machine Learning model gets lazy and assigns all the weightage to attendance[cite: 705].
* [cite_start]**Data Source Limitations:** The Kaggle Dataset does not exactly document how the data was collected or obtained[cite: 706]. [cite_start]We do not know where the data came from, and knowing which country or city will be important as we can determine whether it’s fair to use it as a general model for all students[cite: 707]. [cite_start]The grade level of the students was also not specified[cite: 708].

## [cite_start]AREAS OF FURTHER STUDY [cite: 710]
* [cite_start]We could analyze how each factor correlates with the other, as that would give more insight into how our model is affected by multicollinearity[cite: 727].
* [cite_start]We could analyze our models with Explainable AI Methods such as Lime and Shap, to gain better insight on how our models work[cite: 729].
* [cite_start]It will be good to explore different models to train our dataset on, and compare with the model we have now in order to see which algorithm accurately models the real world[cite: 730].
* A survey form could be created for students to answer based on these performance factors. [cite_start]We can then run inference on the answers given, in order to give an output of an exam score based on the student’s answer[cite: 731, 732].

## GROUP ASSESSMENT
We worked equally I would say, my partner used Statistical Analysis to find correlation and I used Machine Learning. [cite_start]The contribution will be 50/50[cite: 732, 733].

## REFERENCES
1.  “Categorical Data Encoding Techniques in Machine Learning.” [cite_start]GeeksforGeeks , GeeksforGeeks, 18 Sept. 2025, www.geeksforgeeks.org/machine-learning/categorical-data-encoding-techniques-in-machine-learning/[cite: 733].
2.  “Collinearity.” [cite_start]Wikipedia , Wikimedia Foundation, 19 July 2025, en.wikipedia.org/wiki/Collinearity[cite: 734].
3.  Comment, et al. “Cross Validation in Machine Learning.” [cite_start]GeeksforGeeks , 29 Oct. 2025, www.geeksforgeeks.org/machine-learning/cross-validation-machine-learning/[cite: 734, 735].
4.  Ng., Lai. “Student Performance Factors.” [cite_start]Kaggle , 21 Aug. 2025, www.kaggle.com/datasets/lainguyn123/student-performance-factors[cite: 735].
5.  “One Hot Encoding in Machine Learning.” [cite_start]GeeksforGeeks , 11 July 2025, www.geeksforgeeks.org/machine-learning/ml-...[cite: 736].
