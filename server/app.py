from markupsafe import escape
from flask import Flask

app = Flask(__name__)

@app.route('/')
def index():
    return 'index page'
