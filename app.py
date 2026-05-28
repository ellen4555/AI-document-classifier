import pickle
import tkinter as tk
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

# Training data
documents = [
    
    # Sports
    "The football match was exciting",
    "The cricket team won the tournament",
    "Basketball players trained hard",
    "The soccer championship starts tomorrow",
    "Football players scored many goals",
    "The team practiced for the match",
    "Sports competitions are exciting",
    "Cricket fans celebrated the victory",

    # Technology
    "Artificial intelligence is changing technology",
    "Python is popular for machine learning",
    "New software is released today",
    "Computers are becoming faster",
    "Java is a programming language",
    "Python and Java are used in software development",
    "Web development uses programming languages",
    "Artificial intelligence uses computer programs",

    
    # Education
    "Students are preparing for final exams",
    "Teachers are conducting online classes",
    "The university announced new admissions",
    "School annual exams begin next month",
    "Students are studying in the classroom",
    "Examinations will start next week",
    "College students attended lectures",
    "Teachers are teaching mathematics",

   
    # Health
    "Doctors recommend regular exercise",
    "Healthy food improves immunity",
    "The hospital opened a new department",
    "Vaccination helps prevent diseases",
    "Patients visited the medical clinic",
    "Regular walking improves health",
    "Doctors advised people to drink more water",
    "Medical treatment was successful",


    # Entertainment
    "The new movie became a blockbuster",
    "The actor won an award",
    "Fans enjoyed the music concert",
    "The television show gained popularity",
    "People watched the comedy film",
    "The singer released a new album",
    "The audience enjoyed the performance",
    "The cinema was crowded during the weekend",



]
categories = [
    # Sports
    "Sports",
    "Sports",
    "Sports",
    "Sports",
    "Sports",
    "Sports",
    "Sports",
    "Sports",


    # Technology
    "Technology",
    "Technology",
    "Technology",
    "Technology",
    "Technology",
    "Technology",
    "Technology",
    "Technology",

    # Education
    "Education",
    "Education",
    "Education",
    "Education",
    "Education",
    "Education",
    "Education",
    "Education",
    

    # Health
    "Health",
    "Health",
    "Health",
    "Health",
    "Health",
    "Health",
    "Health",
    "Health",

    # Entertainment
    "Entertainment",
    "Entertainment",
    "Entertainment",
    "Entertainment",
    "Entertainment",
    "Entertainment",
    "Entertainment",
    "Entertainment",
]
    
# Convert text into numbers
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(documents)

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X,
    categories,
    test_size=0.2,
    random_state=42
)

# Train model
model = MultinomialNB()
model.fit(X_train, y_train)

# Test accuracy
y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
# Save model
pickle.dump(model, open("model.pkl", "wb"))

# Save vectorizer
pickle.dump(vectorizer, open("vectorizer.pkl", "wb"))

# Prediction function
def predict_category():
    text = entry.get()

    if text.strip() == "":
        result_label.config(text="Please enter some text")
        return

    test_X = vectorizer.transform([text])

    prediction = model.predict(test_X)

    result_label.config(
        text="Predicted Category: " + prediction[0]
    )

# GUI Window
window = tk.Tk()
window.title("AI Document Classifier")
window.geometry("500x350")
window.configure(bg="#dbeafe")

# Title
title = tk.Label(
    window,
    text="AI Document Classifier",
    font=("Arial", 20, "bold"),
    bg="#dbeafe",
    fg="#1e3a8a"
)
title.pack(pady=20)

# Instruction
instruction = tk.Label(
    window,
    text="Enter a document or sentence",
    font=("Arial", 12),
    bg="#dbeafe"
)
instruction.pack()

# Input box
entry = tk.Entry(
    window,
    width=45,
    font=("Arial", 14)
)
entry.pack(pady=15)

# Predict button
predict_button = tk.Button(
    window,
    text="Predict Category",
    font=("Arial", 12, "bold"),
    bg="#2563eb",
    fg="white",
    padx=10,
    pady=5,
    command=predict_category
)
predict_button.pack(pady=10)

# Result label
result_label = tk.Label(
    window,
    text="Predicted Category: ",
    font=("Arial", 15, "bold"),
    bg="#dbeafe",
    fg="#111827"
)
result_label.pack(pady=20)



accuracy_label = tk.Label(
    window,
    text=f"Model Accuracy: {accuracy*100:.2f}%",
    font=("Arial", 12, "bold"),
    bg="#dbeafe",
    fg="green"
)

accuracy_label.pack()

window.mainloop()