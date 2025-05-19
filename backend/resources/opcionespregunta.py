#resources/opcionepregunta.py

from flask import jsonify
from flask_restful import Resource
from db import mongo

class OpcionesPreguntaResource(Resource):
    def get(self):
        opciones = mongo.db.opcionespregunta.find()
        result = [
            {
                "_id": str(opcion["_id"]),
                "opcionPregunta": opcion["opcionPregunta"],
                "idPregunta": str(opcion["idPregunta"]),
                "peso": opcion["peso"]
            }
            for opcion in opciones
        ]
        return jsonify(result)
