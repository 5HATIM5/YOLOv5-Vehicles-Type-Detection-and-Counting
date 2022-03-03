from flask import Flask
from flask_cors import CORS

app = Flask(__name__)
CORS(app)
app.secret_key = "hhfsdfhs00390dsafjsdafkh30940"

from app import views
