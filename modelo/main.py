from model.train import train_model
from model.predict import predict_next_question

def main():
    while True:
        print("\n--- Menú Principal ---")
        print("1. Entrenar modelo")
        print("2. Predecir siguiente pregunta")
        print("3. Salir")

        opcion = input("Elige una opción: ").strip()

        if opcion == "1":
            print("Entrenando modelo...")
            train_model()
            print("Entrenamiento finalizado.")
        elif opcion == "2":
            id_alumno = input("Introduce el ID del alumno: ").strip()
            resultado = predict_next_question(id_alumno)

            if "error" in resultado:
                print(resultado["error"])
            else:
                print("\n--- Pregunta Sugerida ---")
                print(f"ID Pregunta: {resultado['idPregunta']}")
                print(f"Texto: {resultado['pregunta']}")
                print(f"Ámbitos: {resultado['ambitos']}")
                print(f"Objetivo del alumno: {resultado['objetivo']}")
                print("\nOpciones disponibles:")
                for opcion in resultado["opcionesPregunta"]:
                    print(f"  - {opcion['opcionPregunta']} (Peso: {opcion['peso']})")

        elif opcion == "3":
            print("Saliendo...")
            break
        else:
            print("Opción inválida. Intenta de nuevo.")

if __name__ == "__main__":
    main()
