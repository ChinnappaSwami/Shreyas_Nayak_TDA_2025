import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error, accuracy_score, precision_score, recall_score, f1_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer

nltk.download('vader_lexicon')
sia = SentimentIntensityAnalyzer()

# Load datasets
df = pd.read_excel('Standup_comic_dataset.csv.xlsx')
test_df = pd.read_excel('test_standup.xlsx')
df.dropna(inplace=True)

### Q1 ###
X = df[['view_count', 'commentCount', 'dislike_count']]
y = df['like_count']
model = LinearRegression()
model.fit(X, y)
print("Q1 R²:", model.score(X, y))
n, p = len(y), X.shape[1]
adjusted_r2 = 1 - (1 - model.score(X, y)) * (n - 1) / (n - p - 1)
print("Q1 Adjusted R²:", adjusted_r2)

input_df = pd.DataFrame({
    'view_count': [392658, 454505, 199512, 9028403],
    'commentCount': [189, 143, 93, 7769],
    'dislike_count': [0, 69, 23, 308]
})

print("Q1 Predictions:", model.predict(input_df))

### Q2 ###
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model2 = LinearRegression()
model2.fit(X_train, y_train)
y_pred = model2.predict(X_test)
print("Q2 R²:", r2_score(y_test, y_pred))
print("Q2 MAE:", mean_absolute_error(y_test, y_pred))

### Q3 ###
splits = [0.4, 0.2, 0.1]
train_errors, test_errors = [], []
for s in splits:
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=s, random_state=42)
    model.fit(X_train, y_train)
    train_errors.append(mean_absolute_error(y_train, model.predict(X_train)))
    test_errors.append(mean_absolute_error(y_test, model.predict(X_test)))

plt.plot(['60-40', '80-20', '90-10'], train_errors, label='Train MAE')
plt.plot(['60-40', '80-20', '90-10'], test_errors, label='Test MAE')
plt.title('Train vs Test Error')
plt.legend()
plt.grid(True)
plt.show()

### Q4 ###
df['sentiment_score'] = df['video_title'].apply(lambda title: sia.polarity_scores(str(title))['compound'])
corr = df['sentiment_score'].corr(df['like_count'])
print("Q4 Correlation:", corr)
sns.scatterplot(x='sentiment_score', y='like_count', data=df)
plt.title('Sentiment Score vs Like Count (NLTK VADER)')
plt.xlabel('Sentiment Score')
plt.ylabel('Like Count')
plt.grid(True)
plt.show()

### Q5 ###
X_simple = df[['view_count']]
y_simple = df['like_count']
X_norm = (X_simple - X_simple.mean()) / X_simple.std()
y_norm = (y_simple - y_simple.mean()) / y_simple.std()
m, c = 0.0, 0.0
alpha = 0.01
costs = []
for _ in range(1000):
    y_pred = m * X_norm['view_count'] + c
    error = y_pred - y_norm
    cost = (error ** 2).mean()
    costs.append(cost)
    m -= alpha * (2 * (error * X_norm['view_count']).mean())
    c -= alpha * (2 * error.mean())
plt.plot(costs)
plt.title('Gradient Descent Cost Convergence')
plt.xlabel('Iterations')
plt.ylabel('Cost')
plt.grid(True)
plt.show()

### Q6 ###
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
model_scaled = LinearRegression()
model_scaled.fit(X_scaled, y)
print("Q6 R² with Standardization:", model_scaled.score(X_scaled, y))

### Q7 ###
df['title_length'] = df['video_title'].apply(lambda x: len(str(x)))
correlation = df['title_length'].corr(df['like_count'])
print("Q7 Correlation between title_length and like_count:", correlation)

### Q8 ###
median_like = df['like_count'].median()
df['is_hit'] = (df['like_count'] > median_like).astype(int)
X_cls = df[['view_count', 'commentCount']]
y_cls = df['is_hit']
X_train, X_test, y_train, y_test = train_test_split(X_cls, y_cls, test_size=0.2, random_state=42)
clf = LogisticRegression()
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
print("Q8 Accuracy:", accuracy_score(y_test, y_pred))
print("Q8 Precision:", precision_score(y_test, y_pred))
print("Q8 Recall:", recall_score(y_test, y_pred))
print("Q8 F1 Score:", f1_score(y_test, y_pred))

### Q9 ###
test_df['publishedAt'] = pd.to_datetime(test_df['publishedAt'])
df['publishedAt'] = pd.to_datetime(df['publishedAt'])
df['timestamp'] = df['publishedAt'].astype(np.int64) // 10**9
X_time = df[['timestamp']]
y_time = df['view_count']
model_time = LinearRegression()
model_time.fit(X_time, y_time)

test_df['timestamp'] = test_df['publishedAt'].astype(np.int64) // 10**9
print("Q9 Predicted view_counts:", model_time.predict(test_df[['timestamp']]))

### Q10 ###
tfidf = TfidfVectorizer(stop_words='english')
X_tfidf = tfidf.fit_transform(df['video_title'].astype(str))
model_tfidf = LinearRegression()
model_tfidf.fit(X_tfidf, df['view_count'])
X_test_tfidf = tfidf.transform(test_df['video_title'].astype(str))
print("Q10 TF-IDF Predicted view_counts:", model_tfidf.predict(X_test_tfidf))
