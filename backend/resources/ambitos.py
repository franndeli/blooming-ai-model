#resources/ambitos.py

from flask import jsonify
from flask_restful import Resource
from db import mongo

class AmbitosResource(Resource):
    def get(self):
        ambitos = mongo.db.ambitos.find()
        result = [
            {
                "_id": str(ambito["_id"]),
                "ambito": ambito["ambito"]
            }
            for ambito in ambitos
        ]
        return jsonify(result)
