from flask import Flask, render_template

app = Flask(__name__)

@app.route("/")
def home():
    # Redirect to page1.html
    return render_template('home.html')
@app.route("/TTS")
def TTS():
    # Render page1.html
    return render_template('TTS.html')
@app.route("/menu")
def menu():
    # Render page1.html
    return render_template('menu.html')

@app.route("/STT")
def Stt():
    return render_template('STT.html')

