from flask import Flask, request, jsonify, render_template
from langchain_groq import ChatGroq

app = Flask(__name__)

GROQ_API_KEY = "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"
groq_model = ChatGroq(
    temperature=0.7,
    groq_api_key=GROQ_API_KEY,
    model_name="llama-3.3-70b-versatile"
)

user_sessions = {}


def analyze_mental_state(conversation):
    """Analyzes user's mental state based on conversation history."""
    response = groq_model.invoke(
        f"Based on this conversation history, evaluate the user's mental state briefly: {conversation}"
    )
    return response.content if hasattr(response, "content") else "Unable to analyze mental state."


def generate_progressive_question(conversation):
    """Generates a progressive question."""
    few_shot_examples = [
        "User: I feel stressed about work.\nBot: What specifically about work is stressing you out?",
        "User: I am feeling lonely.\nBot: Have you tried reaching out to close friends or family?",
        "User: I don’t know what to do with my life.\nBot: What are some things that make you feel excited or fulfilled?"
    ]
    prompt = "Here are examples of progressive questions based on user input:\n" + "\n".join(
        few_shot_examples) + f"\nUser's conversation history: {conversation}\nmaintain a natural convo like a therapist medium lenght text quick to read and keep it natural and help them overcome thier problems. no need to mention previous things each time:"

    response = groq_model.invoke(prompt)
    return response.content if hasattr(response, "content") else "I'm not sure what to ask next."


@app.route("/")
def index():
    """Serve the chatbot UI."""
    return render_template("chat.html")


@app.route("/chat", methods=["POST"])
def chat():
    """Handles chat API requests."""
    data = request.json
    user_id = data.get("user_id")
    user_message = data.get("message")

    if not user_id or not user_message:
        return jsonify({"error": "Invalid request, missing user_id or message"}), 400

    if user_id not in user_sessions:
        user_sessions[user_id] = []

    user_sessions[user_id].append(user_message)

    if len(user_sessions[user_id]) % 4 == 0:
        analysis = analyze_mental_state(user_sessions[user_id])
        guidance = groq_model.invoke(f"Provide guidance based on this analysis: {analysis}")
        response_text = guidance.content if hasattr(guidance, "content") else "Guidance unavailable."
        user_sessions[user_id].append(response_text)
        return jsonify({"response": response_text})

    progressive_question = generate_progressive_question(user_sessions[user_id])
    user_sessions[user_id].append(progressive_question)
    return jsonify({"response": progressive_question})


if __name__ == "__main__":
    app.run(debug=True, port=5001)
