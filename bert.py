from flask import Flask, render_template, request, session, jsonify
import tensorflow as tf
import numpy as np
from transformers import TFAutoModel, AutoTokenizer
from langchain_groq import ChatGroq
import os  # Import the os module

app = Flask(__name__)
app.secret_key = "super secret key"  # Required for session management

# Define class labels
class_labels = ['Normal', 'Suicidal', 'Borderline Personality Disorder', 'Bipolar', 'Schizotypal PD']

# Load tokenizer
try:
    # Use relative paths for portability.  Assume 'tokenizer' directory is in the same directory as app.py
    tokenizer_path = os.path.join(os.path.dirname(__file__), "tokenizer", "content", "tokenizer")
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
except Exception as e:
    print(f"Error loading tokenizer: {e}")
    tokenizer = None  # Set to None if loading fails

# Load pre-trained BERT model and classifierpip freeze > requirements.txt
try:
    bert_model = TFAutoModel.from_pretrained("bert-base-uncased")

    class BERTForClassification(tf.keras.Model):
        def __init__(self, bert_model, num_classes):
            super().__init__()
            self.bert = bert_model
            self.fc = tf.keras.layers.Dense(num_classes, activation='softmax')

        def call(self, inputs):
            x = self.bert(inputs)[1]
            return self.fc(x)

    classifier = BERTForClassification(bert_model, num_classes=len(class_labels))
    classifier(tf.ones((1, 128), dtype=tf.int32))  # Dummy input to build the model

    # Use relative path for weights. Assume 'bert_classifier_weights.h5' is in the same directory as app.py
    weights_path = os.path.join(os.path.dirname(__file__), "bert_classifier_weights.h5")
    classifier.load_weights(weights_path)

except Exception as e:
    print(f"Error loading model: {e}")
    bert_model = None
    classifier = None

# Load Groq model
try:
    GROQ_API_KEY = "KEY"  # Replace with actual API key
    groq_model = ChatGroq(
        temperature=0.9,
        groq_api_key=GROQ_API_KEY,
        model_name="llama-3.3-70b-versatile"  # Model for counseling
    )
except Exception as e:
    print(f"Error loading Groq model: {e}")
    groq_model = None

# Few-shot examples for customized counseling
FEW_SHOT_EXAMPLES = {
    "Suicidal": [
        ("I'm feeling hopeless and don't want to continue.",
         "I'm really sorry you're feeling this way. You're not alone, and there are people who care about you. Please reach out to a trusted friend or a professional. You deserve support."),
        ("I can't handle this pain anymore.",
         "I understand that things feel overwhelming. It’s important to talk to someone who can help. Please consider reaching out to a crisis helpline or a therapist."),
    ],
    "Borderline Personality Disorder": [
        ("I feel like everyone is abandoning me.",
         "It’s common for people with BPD to experience intense fears of abandonment. Practicing mindfulness and engaging in DBT (Dialectical Behavior Therapy) can help manage these feelings."),
        ("My emotions feel out of control.",
         "Emotional regulation can be challenging. Try grounding exercises and consider seeking therapy to develop coping strategies."),
    ],
    "Bipolar": [
        ("I have extreme mood swings that I can't control.",
         "Bipolar disorder can be tough, but mood tracking and medication management can help. Have you considered consulting a psychiatrist?"),
        ("Sometimes I feel unstoppable, and then I crash completely.",
         "Managing bipolar disorder often involves stabilizing routines and professional guidance. Consider therapy and support groups."),
    ],
    "Schizotypal PD": [
        ("I feel disconnected from reality and see patterns others don’t.",
         "It can be distressing to feel disconnected. Grounding techniques and structured therapy can help bring clarity and manage symptoms."),
        ("People say I behave oddly, but I feel like I understand things differently.",
         "Your perspective is valid. Finding the right support system and professional guidance can help navigate social and cognitive challenges."),
    ]
}

# Prediction function
def predict(text):
    if tokenizer is None or classifier is None:
        print("Tokenizer or classifier not loaded.")
        return None, None
    try:
        inputs = tokenizer(text, padding=True, truncation=True, return_tensors="tf", max_length=128)
        predictions = classifier(inputs)
        predicted_class_idx = np.argmax(predictions.numpy(), axis=1)[0]
        probabilities = predictions.numpy().flatten()
        return class_labels[predicted_class_idx], probabilities
    except Exception as e:
        print(f"Prediction error: {e}")
        return None, None

# Groq Counseling Function with Few-Shot Prompting
def get_counseling_recommendation(condition):
    if groq_model is None:
        return "Groq model not loaded."

    # Select few-shot examples based on predicted class
    examples = FEW_SHOT_EXAMPLES.get(condition, [])

    few_shot_prompt = "You are a mental health counselor. Provide a compassionate, structured response for each case:\n\n"

    for user_input, expected_response in examples:
        few_shot_prompt += f"User: {user_input}\nCounselor: {expected_response}\n\n"

    few_shot_prompt += f"Now, provide advice for someone experiencing similar to but not same as examples provided keep it short 150 words {condition}:\nCounselor:"

    try:
        response = groq_model.invoke(few_shot_prompt)
        return response.content if hasattr(response, "content") else "Error in generating counseling response."
    except Exception as e:
        return f"Error in generating counseling response: {str(e)}"

@app.route("/", methods=["GET"])
def index():
    return render_template("index.html")  # Ensure you create an index.html template

@app.route("/predict", methods=["POST"])
def predict_route():
    user_input = request.form["text"]
    if user_input.strip():
        predicted_class, probabilities = predict(user_input)

        if predicted_class:
            session["predicted_class"] = predicted_class
            session["probabilities"] = probabilities.tolist()  # Convert numpy array to list

            if predicted_class != "Normal":
                session["counseling_response"] = get_counseling_recommendation(predicted_class)
            else:
                session["counseling_response"] = None

            return jsonify({
                "success": True,
                "predicted_class": predicted_class,
                "probabilities": probabilities.tolist()
            })
        else:
            return jsonify({"success": False, "error": "Prediction failed."})
    else:
        return jsonify({"success": False, "error": "Please enter text."})

@app.route("/counseling", methods=["GET"])
def counseling_route():
    if "counseling_response" in session and session["counseling_response"] is not None:
        return jsonify({"counseling_response": session["counseling_response"]})
    else:
        return jsonify({"message": "No counseling recommendation available. Please perform a prediction first."})

if __name__ == "__main__":
    app.run(debug=True)
