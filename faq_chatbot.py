from tkinter import *
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

faq_questions = [
    "What is CodeAlpha?",
    "How long is the internship?",
    "Will I get a certificate?",
    "How many tasks should I complete?",
    "Is GitHub required?"
]

faq_answers = [
    "CodeAlpha is a software development company.",
    "The internship duration is 1 month.",
    "Yes, after successful completion.",
    "Complete at least 2 tasks.",
    "Yes, project code must be uploaded to GitHub."
]

def get_response():
    user_question = entry.get()

    all_questions = faq_questions + [user_question]

    vectorizer = TfidfVectorizer()
    vectors = vectorizer.fit_transform(all_questions)

    similarity = cosine_similarity(vectors[-1], vectors[:-1])

    best_match = similarity.argmax()

    response_label.config(text=faq_answers[best_match])

root = Tk()
root.title("FAQ Chatbot")
root.geometry("500x300")

Label(root, text="Ask a Question").pack(pady=10)

entry = Entry(root, width=50)
entry.pack()

Button(root, text="Get Answer", command=get_response).pack(pady=10)

response_label = Label(root, text="", wraplength=400)
response_label.pack(pady=20)

root.mainloop()
