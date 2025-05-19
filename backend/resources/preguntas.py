#resources/preguntas.py

from flask import jsonify
from flask_restful import Resource
from db import mongo

class PreguntasResource(Resource):
    def get(self):
        preguntas = mongo.db.preguntas.find()
        result = [
            {
                "_id": str(pregunta["_id"]),
                "pregunta": pregunta["pregunta"],
                "ambitos": pregunta["ambitos"]
            }
            for pregunta in preguntas
        ]
        return jsonify(result)
