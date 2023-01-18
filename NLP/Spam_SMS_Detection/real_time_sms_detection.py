import pickle

def classify_sms(sms):
    # Load the saved model
    with open('spam_classifier.pkl', 'rb') as f:
        clf = pickle.load(f)
        
    # Load the saved vectorizer
    with open('vectorizer.pkl', 'rb') as f:
        vectorizer = pickle.load(f)

    # Vectorize the input SMS
    sms_vector = vectorizer.transform([sms])
    
    # Predict the class
    label = clf.predict(sms_vector)
    
    if label[0] == 0:
        return "Not Spam"
    else:
        return "Spam"

# Example usage
new_sms = input("Write the SMS:")
print(classify_sms(new_sms))  # Output: Spam
