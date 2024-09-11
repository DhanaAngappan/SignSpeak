from flask import Flask, render_template, request, redirect, url_for
import os 
from time import time
import subprocess
import firebase_admin
from firebase_admin import credentials, auth, db

app = Flask(__name__)

output1 = ""

cred = credentials.Certificate('credentials.json')
firebase_admin.initialize_app(cred, {
    'databaseURL': 'https://projectdhh-81b8f-default-rtdb.firebaseio.com/'
})
ref = db.reference()

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        email = request.form['email']
        password = request.form['password']
        try:
            user = auth.get_user_by_email(email)
            auth_user = auth.sign_in_with_email_and_password(email, password)
            return redirect(url_for('home'))
        except auth.AuthError as e:
            error_message = e.message
            return render_template('login.html', error=error_message)
    return render_template('login.html')

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        name = request.form['name']
        email = request.form['email']
        password = request.form['password']
        confirm_password = request.form['confirm_password']
        if password == confirm_password:
            try:
                user = auth.create_user(email=email, password=password)
                uid = user.uid
                ref.child('users').child(uid).set({
                    'name': name,
                    'email': email
                })
                return redirect(url_for('home'))
            except auth.AuthError as e:
                error_message = e.message
                return render_template('register.html', error=error_message)
        else:
            error_message = "Passwords do not match"
            return render_template('register.html', error=error_message)
    return render_template('register.html')


@app.route('/')
def index():
    return render_template('index.html')

@app.route('/home')
def home():
    return render_template('home.html')


@app.route('/alphabet')
def alphabet():
    print("Tihs is output1",output1)
    return render_template('alph.html', output1=output1)

@app.route('/video_save', methods=['POST'])
def video_save():
    global output1  # Define output1 as global
    if 'video' in request.files:
        video_file = request.files['video']
        video_path = os.path.join('Static', 'sign.mp4') 
        video_file.save(video_path)
        
        try:
            # Execute the command and capture the output
            result = subprocess.run(["python", "char_recognition.py"], capture_output=True, text=True, check=True)
            temp2 = result.stdout.strip()
            print("Temp2 is", temp2)

            # Assign temp2 to output1
            output1 = temp2
        except subprocess.CalledProcessError as e:
            # Log any errors that occurred during command execution
            print("Error executing number_recognition.py:", e)
            output1 = ""  # Reset output1 on error  
        return redirect(url_for('alphabet') + '#reload')
    else:
        return render_template('home.html') 
    
    
@app.route('/number')
def number():
    print("This is output1:", output1)
    return render_template('num.html', output1=output1)


@app.route('/save_video', methods=['POST'])
def save_video():
    global output1
    if 'video' in request.files:
        video_file = request.files['video']
        video_path = os.path.join('Static', 'sign.mp4')
        video_file.save(video_path)

        try:
            # Execute the command and capture the output
            result = subprocess.run(["python", "number_recognition.py"], capture_output=True, text=True, check=True)
            temp2 = result.stdout.strip()
            print("Temp2 is", temp2)

            # Assign temp2 to output1
            output1 = temp2
        except subprocess.CalledProcessError as e:
            # Log any errors that occurred during command execution
            print("Error executing number_recognition.py:", e)
            output1 = ""  # Reset output1 on error

        # Redirect to the same URL with a hash to force reload
        return redirect(url_for('number') + '#reload')
    else:
        return render_template('home.html')
    
@app.route('/words')
def words():
    print("This is output1:", output1)
    return render_template('words.html', output1=output1)

@app.route('/process_video', methods=['POST'])
def process_video():
    global output1
    if 'video' in request.files:
        video_file = request.files['video']
        video_path = os.path.join('Static', 'sign.mp4')
        video_file.save(video_path)

        try:
            # Execute the command and capture the output
            result = subprocess.run(["python", "words_recogniser.py"], capture_output=True, text=True, check=True)
            temp2 = result.stdout.strip()
            print("Temp2 is", temp2)

            # Assign temp2 to output1
            output1 = temp2
        except subprocess.CalledProcessError as e:
            # Log any errors that occurred during command execution
            print("Error executing number_recognition.py:", e)
            output1 = ""  # Reset output1 on error

        # Redirect to the same URL with a hash to force reload
        return redirect(url_for('words') + '#reload')
    else:
        return render_template('home.html')

    

@app.route('/lesson')
def lesson():
    return render_template('lesson.html')

@app.route('/Contact')
def contact():
    return render_template('Contact.html')

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/index')
def logOut():
    return render_template('index.html')
        

if __name__ == '__main__':
    app.run(debug=True)



