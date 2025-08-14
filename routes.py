from flask import Blueprint, jsonify
from extensions import db

routes = Blueprint("routes", __name__)

@routes.route("/")
def home():
    return jsonify({"status": "API rodando!"})
