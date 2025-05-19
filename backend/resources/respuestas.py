#resources/respuestas.py

from flask import jsonify
from flask_restful import Resource
from db import mongo

class RespuestasResource(Resource):
    def get(self):
        respuestas = mongo.db.respuestas.aggregate([
            {
                "$lookup": {
                    "from": "alumnos",
                    "localField": "idAlumno",
                    "foreignField": "_id",
                    "as": "alumno"
                }
            },
            {
                "$lookup": {
                    "from": "preguntas",
                    "localField": "idPregunta",
                    "foreignField": "_id",
                    "as": "pregunta"
                }
            },
            {
                "$lookup": {
                    "from": "opcionespregunta",
                    "localField": "idOpcionPregunta",
                    "foreignField": "_id",
                    "as": "opcion"
                }
            },
            {"$unwind": "$alumno"},
            {"$unwind": "$pregunta"},
            {"$unwind": "$opcion"}
        ])

        result = [
            {
                "_id": str(respuesta["_id"]),
                "idAlumno": str(respuesta["idAlumno"]),
                "NombreAlumno": respuesta["alumno"]["nombre"],
                "idPregunta": str(respuesta["idPregunta"]),
                "pregunta": respuesta["pregunta"]["pregunta"],
                "idOpcionPregunta": str(respuesta["idOpcionPregunta"]),
                "opcionPregunta": respuesta["opcion"]["opcionPregunta"],
                "fechaRespuesta": respuesta["fechaRespuesta"].isoformat()
            }
            for respuesta in respuestas
        ]
        return jsonify(result)
