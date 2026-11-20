import re
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

print("=" * 50)
print("       PHISHING EMAIL DETECTION MODEL")
print("=" * 50)

# 1. Sample Email Dataset
data = {
    "email": [
        "Congratulations! You won a free prize. Click this link now.",
        "Your bank account is blocked. Verify your password immediately.",
        "Urgent! Update your account details to avoid suspension.",
        "You have won a lottery. Claim your reward now.",
        "Click here to confirm your bank account information.",
        "Your account is suspended. Login immediately.",
        "Free gift waiting for you. Claim your prize.",
        "Verify your identity using this suspicious link.",
        "Your password has expired. Update it now.",
        "You won a cash reward. Click here to collect.",
        "Urgent payment required. Verify your account.",
        "Your credit card is blocked. Confirm your details.",
        "Click this link to receive your free bonus.",
        "Your account will close today. Login now.",
        "Claim your free voucher by clicking here.",
        "Security alert! Verify your password immediately.",

        "Hello, your meeting is scheduled for tomorrow.",
        "Thank you for your purchase. Your order is confirmed.",
        "Your assignment submission was successful.",
        "Please find the attached meeting schedule.",
        "Your monthly bank statement is available.",
        "The college event starts at 10 AM tomorrow.",
        "Your appointment is confirmed for Monday.",
        "Thank you for attending our online meeting.",
        "Your payment receipt is attached.",
        "Please review the project report.",
        "The examination timetable is available.",
        "Your registration has been completed successfully.",
        "The team meeting will begin at 2 PM.",
        "Your delivery is scheduled for tomorrow.",
        "Welcome to our official newsletter.",
        "Your application has been received successfully."
    ],

    "label": [
        1, 1, 1, 1, 1, 1, 1, 1,
        1, 1, 1, 1, 1, 1, 1, 1,
        0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0
    ]
}

df = pd.DataFrame(data)

# 2. Extract Email Features
suspicious_words = [
    "urgent", "verify", "password", "blocked",
    "suspended", "free", "prize", "claim",
    "login", "account", "click", "reward"
]

def extract_features(email):
    email = email.lower()

    url_count = len(re.findall(r'https?://|www\.', email))

    keyword_count = sum(
        email.count(word) for word in suspicious_words
    )

    https_count = email.count("https://")

    features = (
        email
        + " URL_COUNT_" + str(url_count)
        + " KEYWORD_COUNT_" + str(keyword_count)
        + " HTTPS_COUNT_" + str(https_count)
    )

    return features

df["features"] = df["email"].apply(extract_features)

# 3. Split Dataset
X = df["features"]
y = df["label"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.25,
    random_state=42,
    stratify=y
)

# 4. Convert Text into Numerical Features
vectorizer = TfidfVectorizer()

X_train_vector = vectorizer.fit_transform(X_train)
X_test_vector = vectorizer.transform(X_test)

# 5. Train Machine Learning Model
model = LogisticRegression()

model.fit(X_train_vector, y_train)

# 6. Predict Test Emails
y_pred = model.predict(X_test_vector)

accuracy = accuracy_score(y_test, y_pred)

print("\nModel Training Completed!")

# 7. Display Accuracy
print("\nModel Accuracy:",
      round(accuracy * 100, 2), "%")

# 8. Display Confusion Matrix
cm = confusion_matrix(y_test, y_pred)

print("\nConfusion Matrix:")
print(cm)

print("\nClassification Report:")
print(classification_report(
    y_test,
    y_pred,
    target_names=["Safe", "Phishing"],
    zero_division=0
))

# 9. Test a New Email
print("\n" + "=" * 50)
print("          TEST YOUR EMAIL")
print("=" * 50)

new_email = input("\nEnter an email message: ")

new_features = extract_features(new_email)

new_vector = vectorizer.transform([new_features])

prediction = model.predict(new_vector)

if prediction[0] == 1:
    print("\nResult: PHISHING EMAIL")
else:
    print("\nResult: SAFE EMAIL")

print("\nScan Completed!")