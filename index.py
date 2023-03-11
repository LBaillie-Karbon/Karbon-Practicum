from flask import Flask, request, jsonify
from .models import run_model
import logging

logging.basicConfig(level=logging.DEBUG)
logging.debug("This will get logged")

app = Flask(__name__)

priority_scores, trained_models = run_model()
logging.info(trained_models)

# Define your API endpoint
@app.route("/")
def hello_world():
    return "Hello!"


@app.route("/priority_scores/<tenant_perma_key>/<perma_key>", methods=["GET"])
def get_priority_scores(tenant_perma_key, perma_key):
    logging.info(
        f"Attempted to retrieve priority score for email perma key: {perma_key}, tenant_perma_key: {tenant_perma_key}"
    )

    print(priority_scores.head())
    print(trained_models.head())

    # Check if the perma key exists
    try:
        if priority_scores.get(perma_key):
            logging.debug(f"The perma key, {perma_key}, was found")
            return jsonify(priority_scores.get(perma_key))
    except Exception as e:
        logging.error(e)


if __name__ == "__main__":
    app.run(debug=True)
