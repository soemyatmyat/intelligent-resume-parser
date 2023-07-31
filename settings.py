# systems
import os
from dotenv import load_dotenv

# Load Environment variables
basedir = os.path.abspath(os.path.dirname(__file__))
load_dotenv(os.path.join(basedir, '.env')) # take environment variables from .env.

# Custom
NAME= "intelligent-resume-parser"
URL_PRE = "https://"
ALLOWED_FILE_TYPES = [".pdf", ".doc", ".docx"]
UNZIP_TO_FLD = "resumes" 
BASEDIR_APP=os.path.abspath(os.path.dirname(__file__))
SECRET_KEY=os.environ.get("SECRET_KEY") # https://flask.palletsprojects.com/en/1.1.x/api/#sessions
UPLOAD_FOLDER=os.environ.get("UPLOAD_FOLDER")