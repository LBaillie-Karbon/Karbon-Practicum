from flask import Flask, request, jsonify
from .models import train_model
import logging

logging.basicConfig(level=logging.DEBUG)
logging.debug('This will get logged')

app = Flask(__name__)

priority_scores = train_model()
logging.info(priority_scores)

# Define your API endpoint
@app.route("/")
def hello_world():
    return "Hello!"

@app.route('/priority_scores/<tenant_perma_key>/<perma_key>', methods=['GET'])
def get_priority_scores(tenant_perma_key, perma_key):
    logging.info(f"Attempted to retrieve priority score for perma key : {perma_key}")

    # Check if the perma key exists
    if not priority_scores.get(perma_key):
        logging.debug(f"The perma key, {perma_key}, was not found")
        return -1    

    return jsonify(priority_scores.get(perma_key))

if __name__ == '__main__':
    app.run(debug=True)