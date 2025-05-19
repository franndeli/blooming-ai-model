from flask import Flask
from flask_restful import Api
from flask_pymongo import PyMongo
from resources.alumnos import AlumnosResource, AlumnoByIdResource
from resources.preguntas import PreguntasResource
from resources.ambitos import AmbitosResource
from resources.objetivos import ObjetivosResource
from resources.opcionespregunta import OpcionesPreguntaResource
from resources.respuestas import RespuestasResource

app = Flask(__name__)
app.config["MONGO_URI"] = "mongodb://localhost:27017/tfg_database"

mongo = PyMongo()
mongo.init_app(app)

api = Api(app)

# Rutas
api.add_resource(AlumnosResource, '/alumnos')
api.add_resource(AlumnoByIdResource, '/alumnos/<string:alumno_id>')
api.add_resource(PreguntasResource, '/preguntas')
api.add_resource(AmbitosResource, '/ambitos')
api.add_resource(ObjetivosResource, '/objetivos')
api.add_resource(OpcionesPreguntaResource, '/opciones_pregunta')
api.add_resource(RespuestasResource, '/respuestas')

if __name__ == '__main__':
    app.run(debug=True)
