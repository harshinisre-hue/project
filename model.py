import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
import joblib
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt 
import seaborn as sns
from preprocessor import load_data

print("hi")
reviews = load_data("IMDB_DATASET.csv")
X = reviews["cleaned_review"]
y = reviews["sentiment"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)

vectorizer = TfidfVectorizer()
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)


print("hi")
model = LinearSVC(dual=False, max_iter=10000)
model.fit(X_train_vec, y_train)

predictions = model.predict(X_test_vec)
print("Accuracy:", accuracy_score(y_test, predictions))

joblib.dump(model, "imdb_model.pkl")
joblib.dump(vectorizer, "tfidf_vectorizer.pkl")

print("HI")
cm = confusion_matrix(y_test, predictions)
plt.figure(figsize =(6,6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False,
            xticklabels=['postive', 'negative'],
            yticklabels=['positive', 'negative'])
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.title('Confusion Matrix for Sentiment Classifier')
plt.suptitle(f"Accuracy: {accuracy_score(y_test, predictions)}")
plt.savefig('confusionmatrix_imdb.png')