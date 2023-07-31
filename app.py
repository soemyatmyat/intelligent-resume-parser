from flask import Flask, render_template, request, url_for, flash, redirect
from werkzeug.exceptions import abort 
import logging

#### Blueprint Import ###
from intelligentresumeparser.main import intResumeParserBp # import Intelligent Resume Parser Module

app = Flask(__name__)
app.config.from_pyfile("settings.py") # load environment variables 

logging.basicConfig(filename="intelligent-resume-parser.boringisgood.log", encoding="utf-8", level=logging.DEBUG)

app.register_blueprint(intResumeParserBp) # register Bp

# Main Page Static
@app.route("/")
def index():
  return render_template('index.html')	

''' 
Custom Errors Pages
'''
# 404 Not Found: You Fuck up
@app.errorhandler(404) 
def not_found_error(error):
  logging.error("Page not found: %s", (request.path, error))
  return render_template("404.html"), 404

# 500 Internal Server Error: I fuck up
@app.errorhandler(500) 
def internal_error(error):
  logging.error("Server Error: %s", (error))
  return render_template("500.html"), 500	

# General HTTP Errors
@app.errorhandler(Exception)
def unhandled_exception(error):
    logging.error("Unhandled Exception: %s", (error))
    return render_template("500.html"), 500

if __name__ == "__main__":
   app.run(debug=False, host='0.0.0.0', port=8000)