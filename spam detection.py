import pandas as pd 
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score,confusion_matrix,classification_report 

data=pd.read_csv(r"c:\Users\admin\Desktop\dataset of pythons\spam detection project\spam.csv",encoding="latin-1")
print(data.describe())
print(data.head(10))
print(data.columns)
print(data.isnull().sum())
data=data[['v1','v2']]
data.columns=['label','message']
print(data.isnull().sum())

#convert labels to numeric 
data['label']=data['label'].map({
    'spam':1,
    'ham':0
    })
# input and output
x=data['message']
y=data['label']

# train-test split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=42)

# tf-lDF Vectorization
vectorizer=TfidfVectorizer(stop_words='english',max_df=0.7)
x_train_vec=vectorizer.fit_transform(x_train)
x_test_vec=vectorizer.transform(x_test)

# train model
model=MultinomialNB()
model.fit(x_train_vec,y_train)
import pickle 
pickle.dump(model,open("model.pkl","wb"))
pickle.dump(vectorizer,open("vectorizer.pkl","wb"))
print("Model and vectorizer saved successfully")


# prediction
y_pred=model.predict(x_test_vec)

#evaluation
accuracy=accuracy_score(y_test,y_pred)
print(f"Accuracy:{accuracy*100:.2f}%")
print("\nConfusion MAtrix:\n",confusion_matrix(y_test,y_pred))
print("\nClassification report:\n",classification_report(y_test,y_pred))

# custom message testing
def check_spam(text):
    vec=vectorizer.transform([text])
    result=model.predict(vec)
    print("Spam" if result[0]==1 else "Not Spam")
check_spam("Win cash now!!")
check_spam("Hi,are you free today")










