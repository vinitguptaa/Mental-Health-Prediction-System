from flask import Flask, render_template, request
import joblib
import numpy as np

app = Flask(__name__)

# Load trained model
model = joblib.load("model.pkl")


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    try:
        # Get form data
        gender = int(request.form["gender"])
        age = float(request.form["age"])
        academic_pressure = float(request.form["academic_pressure"])
        work_pressure = float(request.form["work_pressure"])
        cgpa = float(request.form["cgpa"])
        study_satisfaction = float(request.form["study_satisfaction"])
        job_satisfaction = float(request.form["job_satisfaction"])
        sleep_duration = int(request.form["sleep_duration"])
        dietary_habits = int(request.form["dietary_habits"])
        suicidal_thoughts = int(request.form["suicidal_thoughts"])
        work_study_hours = float(request.form["work_study_hours"])
        financial_stress = float(request.form["financial_stress"])
        family_history = int(request.form["family_history"])

        # Final input for model
        features = np.array([[
            gender,
            age,
            academic_pressure,
            work_pressure,
            cgpa,
            study_satisfaction,
            job_satisfaction,
            sleep_duration,
            dietary_habits,
            suicidal_thoughts,
            work_study_hours,
            financial_stress,
            family_history
        ]])

        # ML Prediction
        prediction = model.predict(features)[0]

        # Rule-based score logic for real-world better prediction
        score = 0

        if academic_pressure >= 8:
            score += 2

        if work_pressure >= 8:
            score += 2

        if financial_stress >= 8:
            score += 2

        if sleep_duration == 0:   # Less than 5 hours
            score += 2

        if suicidal_thoughts == 1:
            score += 5

        # Final Result based on score
        if score <= 4:
            result = "Low Mental Wellness Risk"
            suggestion = "You are doing well. Maintain balance in your routine and stay connected with people."

        elif score <= 8:
            result = "Moderate Mental Wellness Risk"
            suggestion = "You may be experiencing some stress. Improve sleep, reduce overload, and take regular breaks."

        else:
            result = "High Mental Wellness Risk"
            suggestion = "Please focus on your mental well-being. Talk to a mentor, counselor, or someone you trust."

        return render_template(
            "result.html",
            prediction=result,
            suggestion=suggestion
        )

    except Exception as e:
        return f"Error: {str(e)}"


@app.route("/recommendation")
def recommendation():
    return render_template("recommendation.html")


@app.route("/chatbot", methods=["GET", "POST"])
def chatbot():
    bot_reply = ""

    if request.method == "POST":
        user_message = request.form["user_message"].lower()

        # Normal conversation
        if "hi" in user_message or "hello" in user_message or "hey" in user_message:
            bot_reply = (
                "Hello 😊 I am here for you. "
                "How are you feeling today? You can share anything like stress, exams, sleep, anxiety, or career worries."
            )

        elif "how are you" in user_message:
            bot_reply = (
                "I am doing well and I am here to support you 😊 "
                "Tell me what is bothering you today."
            )

        elif "thank you" in user_message or "thanks" in user_message:
            bot_reply = (
                "You are always welcome 😊 "
                "Remember, asking for help is a strength, not a weakness."
            )

        elif "i feel stressed" in user_message or "stress" in user_message:
            bot_reply = (
                "It is okay to feel stressed sometimes. "
                "Try taking short breaks, proper sleep, and avoid handling everything at once. "
                "Start with one small step today."
            )

        elif "sad" in user_message or "lonely" in user_message:
            bot_reply = (
                "I am sorry you are feeling this way. "
                "Please stay connected with people who care about you. "
                "Difficult times do pass, even if they feel heavy right now."
            )

        elif "anxiety" in user_message or "panic" in user_message:
            bot_reply = (
                "Take a deep breath. Focus on one thing at a time. "
                "Try slow breathing and avoid overthinking future outcomes."
            )

        elif "exam" in user_message or "study" in user_message:
            bot_reply = (
                "Exam pressure is normal. Make a simple study plan, "
                "focus on consistency, and stop comparing yourself with others."
            )

        elif "sleep" in user_message or "tired" in user_message:
            bot_reply = (
                "Sleep is very important for mental health. "
                "Try reducing mobile use before bed and maintain a fixed sleep schedule."
            )

        elif "career" in user_message or "job" in user_message or "placement" in user_message:
            bot_reply = (
                "Career stress is common, especially during placements. "
                "Focus on daily improvement, not instant success. "
                "Your journey is different from others."
            )

        elif "family" in user_message or "parents" in user_message:
            bot_reply = (
                "Family pressure can feel heavy sometimes. "
                "Try honest communication and remember you do not have to carry everything alone."
            )

        elif "relationship" in user_message or "breakup" in user_message:
            bot_reply = (
                "Emotional pain takes time to heal. "
                "Do not blame yourself too much. Healing slowly is still progress."
            )

        elif "suicide" in user_message or "die" in user_message:
            bot_reply = (
                "Please talk to someone immediately—a trusted friend, family member, mentor, or counselor. "
                "You matter, and this difficult moment will pass. You are not alone."
            )

        elif "how to get out" in user_message or "how to solve" in user_message:
            bot_reply = (
                "Start small. Fix sleep, reduce overload, talk to someone you trust, "
                "and stop trying to solve everything in one day. Small consistent steps create big change."
            )

        else:
            bot_reply = (
                "Thank you for sharing 😊 "
                "I am here to listen. Please tell me more about what is bothering you."
            )

    return render_template(
        "chatbot.html",
        bot_reply=bot_reply
    )


if __name__ == "__main__":
    app.run(debug=True)