
import os

# API Key Configuration
OPENAI_API_KEY = ""

# Data Paths
BASE_DIR = os.path.dirname(os.path.dirname(__file__))
DATA_DIR = os.path.join(BASE_DIR, "Data")
COMPANY_A_PATH = os.path.join(DATA_DIR, "CompanyA")
COMPANY_B_PATH = os.path.join(DATA_DIR, "CompanyB")
