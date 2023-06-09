"""Entry point for simple_chat_pdf."""

from flask import Flask
import os
from routes import configure_routes

app = Flask(__name__)
configure_routes(app)

if __name__ == '__main__':
  app.run(debug=True, host='0.0.0.0', port=int(os.environ.get('PORT', 8080)))