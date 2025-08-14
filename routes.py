from flask import Blueprint, render_template
from extensions import db

routes = Blueprint("routes", __name__)

@routes.route("/")
def home():
    return render_template("index.html")
