# Importing essential libraries
from flask import Flask, render_template, request
import pickle
import numpy as np
from chatterbot import ChatBot
from chatterbot.trainers import ListTrainer
from requests import get
from bs4 import BeautifulSoup
import os
from flask import Flask, render_template, request, jsonify

# Load the Random Forest CLassifier model
filename = 'model.pkl'
model = pickle.load(open(filename, 'rb'))

app = Flask(__name__)

bot= ChatBot('ChatBot')

trainer = ListTrainer(bot)

for file in os.listdir('data/'):

    chats = open('data/' + file, 'r').readlines()

    trainer.train(chats)
   
@app.route('/')
def home():
	return render_template('original.html')
    
def hello():
    return render_template('chat.html')

@app.route('/predict', methods=['GET','POST'])
def predict():
    if request.method == 'POST':

        Glucose = float(request.form['Glucose'])
        BloodPressure = float(request.form.get('BloodPressure'))
        SkinThickness = float(request.form.get('SkinThickness'))
        Insulin = float(request.form['Insulin'])
        BMI = float(request.form['BMI'])
        DiabetesPedigreeFunction = float(request.form.get('DiabetesPedigreeFunction'))
        Age = float(request.form['Age'])
        
        
        data = np.array([[Glucose, BloodPressure, SkinThickness, Insulin, BMI, DiabetesPedigreeFunction, Age]])
        my_prediction = model.predict(data)
        
        return render_template('predict.html', prediction=my_prediction)


@app.route("/chat", methods=['GET','POST'])
def ml():
            
    return render_template('chat.html')
        

@app.route("/ask", methods=['POST'])
def ask():

    message = str(request.form['messageText'])

    bot_response = bot.get_response(message)

    while True:

        if bot_response.confidence > 0.1:

            bot_response = str(bot_response)      
            print(bot_response)
            return jsonify({'status':'OK','answer':bot_response})

        elif message == ("bye"):

            bot_response='Hope to see you soon'

            print(bot_response)
            return jsonify({'status':'OK','answer':bot_response})

            break

        else:
        
            try:
                url  = "https://en.wikipedia.org/wiki/"+ message
                page = get(url).text
                soup = BeautifulSoup(page,"html.parser")
                p    = soup.find_all("p")
                return jsonify({'status':'OK','answer':p[1].text})

            except IndexError as error:

                bot_response = 'Sorry i have no idea about that.'
            
                print(bot_response)
                return jsonify({'status':'OK','answer':bot_response})

if __name__ == '__main__':
	app.run(debug="True")

