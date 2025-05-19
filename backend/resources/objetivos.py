#resources/objetivos.py

from flask import jsonify
from flask_restful import Resource
from db import mongo

class ObjetivosResource(Resource):
    def get(self):
        objetivos = mongo.db.objetivos.find()
        result = [
            {
                "_id": str(objetivo["_id"]),
                "objetivo": objetivo["objetivo"]
            }
            for objetivo in objetivos
        ]
        return jsonify(result)
