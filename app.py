from flask import Flask, render_template, request, jsonify, Response
import google.generativeai as genai
import cv2

app = Flask(__name__)

# Your Gemini API key
GEMINI_API_KEY = "AIzaSyCMZaggKNyhuInr8FK6TtN39x-gFHA2aWw"
genai.configure(api_key=GEMINI_API_KEY)

# Save hate speech words to tweet_analysis.txt
HATE_SPEECH_FILE = "tweet_data.txt"
hate_speech_words = {
    "hate"
}
with open(HATE_SPEECH_FILE, "w") as file:
    file.write("\n".join(hate_speech_words))

# Chatbot response function with hate speech detection
def get_bot_response(user_input):
    # Check for hate speech
    contains_hate_speech = any(word in user_input.lower() for word in hate_speech_words)
    if contains_hate_speech:
        return "Your message contains inappropriate language. Please maintain a respectful tone.", None

    # Normal chatbot logic
    try:
        model = genai.GenerativeModel("gemini-1.5-flash")
        chat = model.start_chat(
            history=[
                {"role": "model", "parts": "I’ll respond to your questions like a Hate Speech detector, and ask you to report"},
                {"role": "user", "parts": "Hello"},
                {"role": "model", "parts": "I'm here to help. How can I assist you today?"}
            ]
        )
        response = chat.send_message(user_input)
        return response.text.strip(), None
    except Exception as e:
        print(f"Error: {e}")
        return "Sorry, I'm currently unavailable. Please try again later.", None

# Route for chatbot page
@app.route('/')
def chatbot():
    return render_template('chatbot.html')

# Handle chatbot responses
@app.route('/get_response', methods=['POST'])
def get_response():
    user_input = request.form.get('message')
    response, _ = get_bot_response(user_input)
    return jsonify({"response": response})

# Smile detection function
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
smile_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_smile.xml')

def generate_frames():
    camera = cv2.VideoCapture(0)
    while True:
        success, frame = camera.read()
        if not success:
            break
        else:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
            smile_score = 0  # Default smile score
            
            for (x, y, w, h) in faces:
                roi_gray = gray[y:y+h, x:x+w]
                smiles = smile_cascade.detectMultiScale(roi_gray, scaleFactor=1.8, minNeighbors=20, minSize=(25, 25))
                smile_score = min(len(smiles) * 20, 100)  # Ensure smile score goes up to 100

                # Display smile score on the frame
                label = f"Smile Score: {smile_score}"
                cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

            # Check if smile score is zero and display a prompt to visit the fun content page
            if smile_score == 0:
                prompt_text = "Go to the Fun Content page to lift your mood!"
                cv2.putText(frame, prompt_text, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
    camera.release()

# Route for smile detection page
@app.route('/smile_score')
def smile_score():
    return render_template('smile_score.html')

# Route to stream the video feed
@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

# Route for fun and motivational content page
@app.route('/fun_content')
def fun_content():
    return render_template('fun_content.html')

if __name__ == '__main__':
    app.run(debug=True)
