from flask import Flask, render_template, url_for,request, redirect, url_for
import easyocr
from PIL import Image
import os
import numpy as np
import tensorflow as tf

from flask import Flask, render_template, request
from transformers import MarianMTModel, MarianTokenizer

from keras.models import load_model
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from flask_login import LoginManager, UserMixin,login_user, login_required, logout_user,current_user
from flask_mysqldb import MySQL


from flask_bcrypt import Bcrypt

# app = Flask(__name__)



# Initialize EasyOCR with the language model you need (e.g., 'en' for English)
# reader = easyocr.Reader(['en'])

# @app.route('/TextTranslator')
# def TextTranslator():
#     return render_template('index.html')

# @app.route('/upload', methods=['POST'])
# def upload():
#     if 'file' not in request.files:
#         return render_template('index.html', error='No file part')

#     file = request.files['file']

#     if file.filename == '':
#         return render_template('index.html', error='No selected file')

#     try:
#         image = Image.open(file)
#         result = reader.readtext(image)
#         recognized_text = '\n'.join([text[1] for text in result])

#         # Save the uploaded image to the 'uploads' folder
#         upload_folder = 'uploads'
#         os.makedirs(upload_folder, exist_ok=True)
#         file_path = os.path.join(upload_folder, file.filename)
#         image.save(file_path)

#         return render_template('result.html', image_path=file_path, text=recognized_text)
#     except Exception as e:
#         return render_template('index.html', error=f'Error processing image: {e}')

# if __name__ == '__main__':
#     app.run(debug=True)

# TextSummerizer code

from flask import Flask, render_template
from flask import request as req
import requests
# from requests.api import request

app = Flask(__name__)

app.secret_key="4657iybhj89fvijl899uibhn"

app.config['MYSQL_HOST']="localhost"
app.config['MYSQL_USER']="root"
app.config['MYSQL_PASSWORD']="man0045chau"
app.config['MYSQL_DB']="flask_database"
# app.config['MYSQL_PORT'] = 3306 
# app.config['MYSQL_UNIX_SOCKET'] = None
# app.config['MYSQL_CONNECT_TIMEOUT'] = 20
# app.config['MYSQL_READ_DEFAULT_FILE'] = None
# app.config['MYSQL_USE_UNICODE'] = False 
# app.config['MYSQL_CHARSET'] = 'utf8mb4'  # Set to the desired character set
# app.config['MYSQL_SQL_MODE'] = None  # Set to the desired SQL mode or None
# app.config['MYSQL_CURSORCLASS'] = None  



mysql= MySQL(app)
login_manage=LoginManager()

login_manage.init_app(app)

bcrypt = Bcrypt(app)

##User load
@login_manage.user_loader
def load_user(user_id):
 return User.get(user_id)

class User(UserMixin):
 def __init__(self, user_id, name, email):
  self.id = user_id
  self.name = name
  self.email = email

 @staticmethod
 def get(user_id):
  cursor = mysql.connection.cursor()
  cursor.execute('SELECT id, name, email,password from users where id = %s', (user_id,))
  result = cursor.fetchone()
  cursor.close()
  if result:
   return User(user_id, result[0], result[1])
  



@app.route('/login', methods=['GET','POST'])
def login():
 if request.method == 'POST':
  email = request.form['email']
  password = request.form['password']

  hashed_password = bcrypt.generate_password_hash(password).decode('utf-8')
  

  cursor = mysql.connection.cursor()
  cursor.execute('SELECT id, name, email, password from users where email = %s', (email,))
  user_data = cursor.fetchone()
  cursor.close()

  if user_data and bcrypt.check_password_hash(user_data[3], password):
   user=User(user_data[0], user_data[1], user_data[2])
   login_user(user)
   return redirect(url_for('indi'))
  


 return render_template('login.html')

@app.route('/register', methods = ['GET','POST'])

def register():
 if request.method == 'POST':
  name = request.form['name']
  email = request.form['email']
  password = request.form['password']

  hashed_password = bcrypt.generate_password_hash(password).decode('utf-8')
  

  cursor = mysql.connection.cursor()
  cursor.execute('INSERT INTO users (name,email,password) values(%s,%s,%s)', (name,email,hashed_password))
  mysql.connection.commit()
  
  cursor.close()
  return redirect(url_for('login'))
 return render_template('register.html')

@app.route('/indi')
@login_required
def indi():
 return render_template('indi.html')




@app.route('/logout')
@login_required
def logout():
 logout_user()
 return redirect(url_for('india'))

@app.route('/india')
def india():
    return render_template('india.html')  # Replace with the actual template file for 'india'


reader = easyocr.Reader(['en'])
@app.route("/", methods=["GET","POST"])
def Index():
 return render_template("india.html")

@app.route('/TextRecognition')
def TextRecognition():
    return render_template('index.html')

@app.route("/services")
@login_required
def service():
 return render_template("Bar/Navbar.html")
# @app.route('/services')
# def service():
#     if current_user.is_authenticated:
#         return render_template("Bar/Navbar.html")
#     else:
#         return render_template("login.html", message="Please log in to access services.")


@app.route("/Text-Summerizer")
def TextSummerizer():
 return render_template("uimain/index2.html")

@app.route("/Summarize",methods=["GET","POST"])
def Summarize():
 if req.method=="POST":
  API_URL = "https://api-inference.huggingface.co/models/facebook/bart-large-cnn"
  headers = {"Authorization": "Bearer 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"}

  data = req.form["data"]
  maxL = int(req.form["maxL"])
  minL = maxL//4
  def query(payload):

   response = requests.post(API_URL, headers=headers, json=payload)
   return response.json()

  output = query({
   "inputs": data,
   "parameters": {"min_length":minL, "max_length":maxL},
  })[0]
 
  return render_template("uimain/index2.html", result=output["summary_text"])

 else:
   return render_template("uimain/index2.html", result="No data provided")

@app.route('/upload', methods=['POST'])
def upload():
    if 'file' not in request.files:
        return render_template('index.html', error='No file part')

    file = request.files['file']

    if file.filename == '':
        return render_template('index.html', error='No selected file')

    try:
        image = Image.open(file)
        result = reader.readtext(image)
        recognized_text = '\n'.join([text[1] for text in result])

        # Save the uploaded image to the 'uploads' folder
        upload_folder = 'static/uploads'
        os.makedirs(upload_folder, exist_ok=True)
        file_path = os.path.join(upload_folder, file.filename)
        image.save(file_path)

        return render_template('result.html', image_path=file_path, text=recognized_text)
    except Exception as e:
        return render_template('index.html', error=f'Error processing image: {e}')
    

    # Text-translate



# Load the trained model
model = load_model("translation_model.h5")

# Load target tokenizer (replace 'target_tokenizer_path' with the actual path)
target_tokenizer = Tokenizer()
target_tokenizer.fit_on_texts(["<start>", "<end>", "your", "training", "target", "vocabulary", "here"])

# Replace these placeholders with your actual functions and variable values
def encode_sequences(texts, tokenizer, max_length):
    # Placeholder implementation, replace with actual encoding logic
    sequences = tokenizer.texts_to_sequences(texts)
    padded_sequences = pad_sequences(sequences, maxlen=max_length, padding='post')
    return padded_sequences

input_tokenizer = Tokenizer()  # Replace with your actual input tokenizer
max_length_input = 50  # Replace with your actual max length for input sequences
reverse_target_word_index = {"1": "<start>", "2": "your", "3": "training", "4": "target", "5": "vocabulary", "6": "here"}  # Replace with your actual reverse target word index
max_length_target = 60  # Replace with your actual max length for target sequences

def translate_text(input_text):
    # Tokenize and pad the input text
    input_seq = encode_sequences([input_text], input_tokenizer, max_length_input)

    # Print the word_index for debugging
    print("Word Index for '<start>':", target_tokenizer.word_index.get("<start>"))
    print("Word Index for '<end>':", target_tokenizer.word_index.get("<end>"))

    # Initialize the decoder input with a start token
    target_seq = np.zeros((1, 1))  # Initialize target_seq here
    start_token_index = target_tokenizer.word_index.get("<start>")
    if start_token_index is not None:
        target_seq[0, 0] = start_token_index
    else:
        # Handle the case where "<start>" is not in the word_index
        print("Warning: '<start>' not found in target tokenizer word_index. Using a default value.")
        # Use a default value or raise an exception based on your requirement
        # Here, we use 0 as a default value
        target_seq[0, 0] = 0

    stop_condition = False
    decoded_sentence = ""

    # Assuming model is the name of your translation model
    # Modify the following line to match your actual model architecture
    output_tokens, h, c = model.predict([input_seq, target_seq])


    while not stop_condition:
        output_tokens, h, c = model.predict([input_seq, target_seq])
        # Sample a token
        sampled_token_index = np.argmax(output_tokens[0, -1, :])
        sampled_token = reverse_target_word_index.get(str(sampled_token_index), "")

        if sampled_token != "<end>":
            decoded_sentence += " " + sampled_token

        # Exit condition: either hit max length or find stop token
        if sampled_token == "<end>" or len(decoded_sentence.split()) > max_length_target:
            stop_condition = True

        # Update the target sequence with the newly sampled token
        target_seq = np.zeros((1, 1))
        target_seq[0, 0] = sampled_token_index

    return decoded_sentence.strip()


# TextTranslator

# Flask routes
# @app.route('/TextTranslator')
# def TextTranslator():
#     return render_template("impo.html")

# @app.route("/translate", methods=["POST"])
# def translate():
#     input_text = request.form["input_text"]
#     translated_text = translate_text(input_text)
#     return render_template("index.html", input_text=input_text, translated_text=translated_text)

class Translator:
    def __init__(self, target_languages):
        # Load the pre-trained MarianMT model and tokenizer for translation
        self.target_languages = target_languages
        self.models = {lang: MarianMTModel.from_pretrained(f"Helsinki-NLP/opus-mt-en-{lang}") for lang in target_languages}
        self.tokenizers = {lang: MarianTokenizer.from_pretrained(f"Helsinki-NLP/opus-mt-en-{lang}") for lang in target_languages}

    def translate_text(self, input_text, target_language):
        # Tokenize input text and generate translation
        inputs = self.tokenizers[target_language](input_text, return_tensors="pt")
        translation = self.models[target_language].generate(**inputs)

        # Decode the translated text
        translated_text = self.tokenizers[target_language].batch_decode(translation, skip_special_tokens=True)[0]
        return translated_text

# Specify the target languages
target_languages = {'de': 'German', 'fr': 'French', 'es': 'Spanish', 'ga':'Irish', 'it': 'italian'}  # Add or remove languages as needed
translator_instance = Translator(list(target_languages.keys()))

@app.route('/TextTranslator', methods=['GET', 'POST'])
def TextTranslator():
    translated_text = ''
    selected_language = 'de'  # Default target language

    if request.method == 'POST':
        input_text = request.form['input_text']
        selected_language = request.form['target_language']
        translated_text = translator_instance.translate_text(input_text, selected_language)

    return render_template('impo.html', translated_text=translated_text, target_languages=target_languages, selected_language=selected_language)

if __name__ =='__main__':
 app.debug=True
 app.run()