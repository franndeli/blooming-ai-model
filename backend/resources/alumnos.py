#resources/alumnos.py

from flask import jsonify
from flask_restful import Resource
from bson.objectid import ObjectId
from db import mongo

class AlumnosResource(Resource):
    def get(self):
        alumnos = mongo.db.alumnos.find()
        result = [
            {
                "_id": str(alumno["_id"]),
                "nombre": alumno["nombre"],
                "apellidos": alumno["apellidos"],
                "ambitos": alumno["ambitos"],
                "objetivo": alumno.get("objetivo", 4)
            }
            for alumno in alumnos
        ]
        return jsonify(result)

class AlumnoByIdResource(Resource):
    def get(self, alumno_id):
        try:
            alumno = mongo.db.alumnos.find_one({"_id": ObjectId(alumno_id)})
        except Exception:
            return jsonify({"error": "Id inválido"}), 400

        if not alumno:
            return jsonify({"error": "Alumno no encontrado"}), 404

        result = {
            "_id": str(alumno["_id"]),
            "nombre": alumno["nombre"],
            "apellidos": alumno["apellidos"],
            "ambitos": alumno["ambitos"],
            "objetivo": alumno.get("objetivo", 4)
        }
        return jsonify(result)
