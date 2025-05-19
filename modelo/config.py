# config.py
BACKEND_URL = "http://localhost:5000"

RESPUESTAS_ENDPOINT = f"{BACKEND_URL}/respuestas"
PREGUNTAS_ENDPOINT = f"{BACKEND_URL}/preguntas"
ALUMNOS_ENDPOINT = f"{BACKEND_URL}/alumnos"

SQLALCHEMY_DATABASE_URI = "mysql+pymysql://root@localhost/tfg_database"
MODEL_PATH = "model/modelo_final.pkl"

