from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report
import nltk
nltk.download('stopwords')
nltk.download('punkt')
#from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from sklearn.svm import SVC
import json
import numpy 

f = open('dataset.json', encoding="utf8")
data = json.load(f)
texts = [ data[i]['text'] for i in range(len(data)) ]
labels = [ data[i]['class'] for i in range(len(data)) ]

X_train, X_test, y_train, y_test = train_test_split(texts, labels, test_size=0.25, random_state=400)

stop_words_ = set(stopwords.words('turkish'))
vectorizer = CountVectorizer(stop_words=list(stop_words_),  max_features=3500)
X_train_vectorized = vectorizer.fit_transform(X_train)
X_test_vectorized = vectorizer.transform(X_test)


"""
!!!!!!  Accuracy Calculation / MultinomialNB

model_nb = MultinomialNB()
model_nb.fit(X_train_vectorized, y_train)

y_pred_nb = model_nb.predict(X_test_vectorized)

accuracy = accuracy_score(y_test, y_pred)

print("-----NAİVE BAYES-----")
print(f'Accuracy: {accuracy}')
print('Classification Report:\n', classification_report(y_test, y_pred))
"""


svm_model = SVC(C=10)
svm_model.fit(X_train_vectorized, y_train)

"""
!!!!!! Accuracy Calculation / SVM
print("Support Vector Machine Model Results:")
print("-----Support Vector Machine-----")
print(f'Accuracy: {accuracy}')
print('Classification Report:\n', classification_report(y_test, y_pred))
y_pred_svm = svm_model.predict(X_test_vectorized)


train_score = svm_model.score(X_train_vectorized, y_train)
test_score = svm_model.score(X_test_vectorized, y_test)

"""

while True:
    print("Enter a sentence to test (type 'q' and enter for quit): ")
    sentence = input()
    if(sentence == "q"):
        break   

    sentence = [sentence]
    sentence_vectorized = vectorizer.transform(sentence)
    prediction = svm_model.predict(sentence_vectorized)
    print(prediction)
    if prediction == 0:
        print("Not depressed")
    else:
        print("Depressed")