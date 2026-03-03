#!/bin/bash
# Build script for Render.com deployment
pip install -r requirements.txt
python -c "import nltk; nltk.download('wordnet'); nltk.download('omw-1.4')"
