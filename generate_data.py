#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json
import random
from datetime import datetime, timedelta
import numpy as np
from collections import Counter
import logging
import os
from modelo.data.fetch_data import get_preguntas, get_opciones_pregunta

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

DOMAINS = [
    "Familia", "Amigos y relaciones", "Académico", "Entorno en clase",
    "Entorno exterior", "Actividades extraescolares", "Autopercepción y emociones",
    "Relación con profesores", "General"
]

OBJECTIVES = ["Consolidación", "Prevención", "Predeterminado", "Exploración"]

EMOTIONAL_PROFILES = ["Positivo", "Neutro", "Negativo"]
EMOTIONAL_WEIGHTS = {
    "Positivo": 0.4,    # 40% de alumnos
    "Neutro": 0.4,      # 40% de alumnos
    "Negativo": 0.2     # 20% de alumnos
}

N_STUDENTS = 200
RESPONSES_PER_STUDENT = 200
BASE_DATE = datetime(2024, 1, 1)
OUTPUT_DIR = "generated_data"

def generate_objectid():
    return ''.join(random.choices('0123456789abcdef', k=24))

def generate_students(n):
    logger.info(f"Generando {n} alumnos con distribución equitativa de objetivos")
    
    nombres = [
        "Javier", "María", "Carlos", "Ana", "Luis", "Elena", "Pablo", "Sofía", "Miguel", "Laura",
        "Alejandro", "Isabel", "Diego", "Carmen", "Daniel", "Patricia", "Fernando", "Lucía", 
        "Andrés", "Raquel", "Hugo", "Sara", "Álvaro", "Marta", "Manuel", "Clara", "Iván", "Irene", 
        "Adrián", "Paula", "Rubén", "Valeria", "David", "Emma", "Óscar", "Noelia", "Víctor", 
        "Alicia", "Jorge", "Nuria", "Raúl", "Beatriz", "Sergio", "Eva", "Gabriel", "Julia", 
        "Francisco", "Rocío", "Enrique", "Natalia"
    ]
    apellidos = [
        "Fernández Moreno", "García López", "Martínez Sánchez", "González Pérez", 
        "Rodríguez Gómez", "López Martínez", "Sánchez García", "Pérez Rodríguez",
        "Hernández Jiménez", "Ruiz Fernández", "Díaz González", "Moreno Torres", 
        "Muñoz Romero", "Álvarez Ortega", "Romero Navarro", "Alonso Domínguez", 
        "Gutiérrez Castro", "Navarro Delgado", "Torres Ramos", "Domínguez Serrano",
        "Vázquez Vargas", "Castro Molina", "Ortega Rubio", "Delgado Cruz", 
        "Ramos Ibáñez", "Serrano Marín", "Iglesias Soler", "Morales Vidal",
        "Rubio León", "Molina Lozano", "Cruz Peña", "Peña Herrera", "Herrera Aguado",
        "Marín Ríos", "Soler Blanco", "Vidal Campos", "León Santos", "Lozano Pardo",
        "Aguado Sáez", "Ríos Cordero", "Blanco Esteban", "Campos Cabrera", 
        "Santos Gil", "Pardo Méndez", "Sáez Molina", "Cordero Beltrán", 
        "Esteban Suárez", "Cabrera Gallardo", "Gil Cano", "Méndez Lara"
    ]
    
    objectives_list = []
    base_count_obj = n // len(OBJECTIVES)
    remainder_obj = n % len(OBJECTIVES)
    
    for obj in OBJECTIVES:
        objectives_list.extend([obj] * base_count_obj)
    
    if remainder_obj:
        objectives_list.extend(random.sample(OBJECTIVES, remainder_obj))
    
    random.shuffle(objectives_list)
    
    emotional_profiles_list = []
    for _ in range(n):
        profile = random.choices(
            population=list(EMOTIONAL_WEIGHTS.keys()),
            weights=list(EMOTIONAL_WEIGHTS.values()),
            k=1
        )[0]
        emotional_profiles_list.append(profile)
    
    students = []
    student_profiles = []  # Para uso interno durante la generación de datos
    
    for i in range(n):
        _id = {"$oid": generate_objectid()}
        nombre = random.choice(nombres)
        apellido = random.choice(apellidos)
        
        emotional_profile = emotional_profiles_list[i]
        
        ambitos = {}
        for domain in DOMAINS:
            initial_weight = random.randint(30, 70)
            
            if emotional_profile == "Positivo":
                initial_weight = min(initial_weight + random.randint(5, 15), 85)
            elif emotional_profile == "Negativo":
                initial_weight = max(initial_weight - random.randint(5, 20), 20)
            
            ambitos[domain] = {
                "peso": initial_weight,
                "porcentaje": 0,
                "count": 0,
                "history": []
            }
        
        # Crear el objeto student para el JSON final (sin perfil emocional)
        student = {
            "_id": _id,
            "nombre": nombre,
            "apellidos": apellido,
            "ambitos": ambitos,
            "objetivo": objectives_list[i]
        }
        
        # Crear objeto con perfil emocional para uso interno
        student_profile = student.copy()
        student_profile["perfil_emocional"] = emotional_profile
        
        students.append(student)
        student_profiles.append(student_profile)
    
    profile_counter = Counter(emotional_profiles_list)
    logger.info(f"Distribución de perfiles emocionales (interno): {dict(profile_counter)}")
    logger.info(f"Distribución de objetivos: {Counter(objectives_list)}")
    return students, student_profiles

def simulate_response_date(base_date, student_index, response_index, total_students, responses_per_student):
    days_span = 60
    
    position = (student_index * responses_per_student + response_index) / (total_students * responses_per_student)
    base_day = position * days_span
    random_offset = random.uniform(-3, 3)
    day_offset = base_day + random_offset
    
    hour = random.randint(8, 20)
    minute = random.randint(0, 59)
    
    new_date = base_date + timedelta(days=day_offset, hours=hour, minutes=minute)
    return new_date.isoformat() + "Z"

def select_domain_for_objective(student_profile, initial_domains_counter=None, is_first_question=False):
    if is_first_question and initial_domains_counter is not None:
        domain_counts = {d: initial_domains_counter.get(d, 0) for d in DOMAINS}
        selected_domain = min(domain_counts, key=domain_counts.get)
        initial_domains_counter[selected_domain] = initial_domains_counter.get(selected_domain, 0) + 1
        return selected_domain
    
    objective = student_profile["objetivo"]
    domains = student_profile["ambitos"]

    if objective == "Consolidación":
        sorted_domains = sorted(domains.items(), key=lambda item: item[1]["peso"], reverse=True)
        
        if random.random() < 0.9:
            top_domains = sorted_domains[:3]
            weights = [3.0, 1.5, 1.0] if len(top_domains) >= 3 else ([3.0, 1.5] if len(top_domains) >= 2 else [1.0])
            total = sum(weights)
            normalized_weights = [w/total for w in weights[:len(top_domains)]]
            return np.random.choice([d[0] for d in top_domains], p=normalized_weights)
        else:
            return sorted_domains[3][0] if len(sorted_domains) > 3 else sorted_domains[0][0]
    
    elif objective == "Prevención":
        sorted_domains = sorted(domains.items(), key=lambda item: item[1]["peso"])
        
        if random.random() < 0.9:
            bottom_domains = sorted_domains[:3]
            weights = [3.0, 1.5, 1.0] if len(bottom_domains) >= 3 else ([3.0, 1.5] if len(bottom_domains) >= 2 else [1.0])
            total = sum(weights)
            normalized_weights = [w/total for w in weights[:len(bottom_domains)]]
            return np.random.choice([d[0] for d in bottom_domains], p=normalized_weights)
        else:
            return sorted_domains[3][0] if len(sorted_domains) > 3 else sorted_domains[0][0]
    
    elif objective == "Exploración":
        sorted_domains = sorted(domains.items(), key=lambda item: item[1].get("porcentaje", 0))
        
        if random.random() < 0.9:
            least_explored = sorted_domains[:3]
            weights = [3.0, 1.5, 1.0] if len(least_explored) >= 3 else ([3.0, 1.5] if len(least_explored) >= 2 else [1.0])
            total = sum(weights)
            normalized_weights = [w/total for w in weights[:len(least_explored)]]
            return np.random.choice([d[0] for d in least_explored], p=normalized_weights)
        else:
            return sorted_domains[3][0] if len(sorted_domains) > 3 else sorted_domains[0][0]
    
    elif objective == "Predeterminado":
        scores = {}
        max_percentage = max([d.get("porcentaje", 0.001) for d in domains.values()]) + 0.001
        
        for domain, data in domains.items():
            inv_peso = (100 - data["peso"]) / 100
            norm_porcentaje = data.get("porcentaje", 0) / max_percentage
            scores[domain] = (inv_peso * 0.7) + (norm_porcentaje * 0.3)
        
        sorted_domains = sorted(scores.items(), key=lambda x: x[1])
        
        if random.random() < 0.8:
            best_candidates = sorted_domains[:3]
            weights = [3.0, 1.5, 1.0] if len(best_candidates) >= 3 else ([3.0, 1.5] if len(best_candidates) >= 2 else [1.0])
            total = sum(weights)
            normalized_weights = [w/total for w in weights[:len(best_candidates)]]
            return np.random.choice([d[0] for d in best_candidates], p=normalized_weights)
        else:
            return sorted_domains[3][0] if len(sorted_domains) > 3 else sorted_domains[0][0]
    
    else:
        return random.choice(list(domains.keys()))

def select_question(questions, chosen_domain, used_questions=None):
    if used_questions is None:
        used_questions = set()
    
    filtered = []
    filtered_strong = []
    
    for q in questions:
        q_id = str(q["_id"]["$oid"]) if isinstance(q["_id"], dict) else str(q["_id"])
        if q_id in used_questions:
            continue
            
        try:
            q_ambitos = json.loads(q["ambitos"]) if isinstance(q["ambitos"], str) else q["ambitos"]
            q_ambitos = {str(k).strip(): v for k, v in q_ambitos.items() 
                         if k is not None and str(k).strip() and str(k).strip().lower() != "null"}
            
            total_weight = sum(q_ambitos.values())
            
            if chosen_domain in q_ambitos:
                domain_weight_percentage = (q_ambitos[chosen_domain] / total_weight * 100) if total_weight > 0 else 0
                
                if domain_weight_percentage > 70:
                    filtered_strong.append(q)
                else:
                    filtered.append(q)
        except Exception as e:
            logger.warning(f"Error al procesar ambitos de pregunta: {e}")
            continue
    
    if filtered_strong and random.random() < 0.8:
        selected = random.choice(filtered_strong)
        used_questions.add(str(selected["_id"]["$oid"]) if isinstance(selected["_id"], dict) else str(selected["_id"]))
        return selected, used_questions
    
    if filtered:
        selected = random.choice(filtered)
        used_questions.add(str(selected["_id"]["$oid"]) if isinstance(selected["_id"], dict) else str(selected["_id"]))
        return selected, used_questions
    
    available = [q for q in questions if (str(q["_id"]["$oid"]) if isinstance(q["_id"], dict) else str(q["_id"])) not in used_questions]
    if available:
        selected = random.choice(available)
        used_questions.add(str(selected["_id"]["$oid"]) if isinstance(selected["_id"], dict) else str(selected["_id"]))
        return selected, used_questions
    
    logger.info("Todas las preguntas han sido usadas, reiniciando el conjunto usado")
    selected = random.choice(questions)
    used_questions = {str(selected["_id"]["$oid"]) if isinstance(selected["_id"], dict) else str(selected["_id"])}
    return selected, used_questions

def select_option_for_domain(question, question_options, chosen_domain, student_profile):
    q_id = str(question["_id"]["$oid"]) if isinstance(question["_id"], dict) else str(question["_id"])
    options = question_options.get(q_id, [])
    
    if not options:
        logger.warning(f"No se encontraron opciones para la pregunta {q_id}")
        return None
    
    matching_options = [opt for opt in options if opt.get("ambito") == chosen_domain]
    
    if matching_options:
        objective = student_profile["objetivo"]
        emotional_profile = student_profile.get("perfil_emocional", "Neutro")
        
        rand_val = random.random()
        
        positive_options = [opt for opt in matching_options if opt.get("peso", 0) > 5]
        negative_options = [opt for opt in matching_options if opt.get("peso", 0) < -5]
        neutral_options = [opt for opt in matching_options if -5 <= opt.get("peso", 0) <= 5]
        
        if emotional_profile == "Positivo":
            if rand_val < 0.7:
                if positive_options:
                    return random.choice(positive_options)
            elif rand_val < 0.9:
                if neutral_options:
                    return random.choice(neutral_options)
            else:
                if negative_options:
                    return random.choice(negative_options)
                    
        elif emotional_profile == "Negativo":
            if rand_val < 0.6:
                if negative_options:
                    return random.choice(negative_options)
            elif rand_val < 0.85:
                if neutral_options:
                    return random.choice(neutral_options)
            else:
                if positive_options:
                    return random.choice(positive_options)
                    
        else:  # Neutro
            if objective == "Consolidación":
                if rand_val < 0.6 and positive_options:
                    return random.choice(positive_options)
                elif rand_val < 0.9 and neutral_options:
                    return random.choice(neutral_options)
                elif negative_options:
                    return random.choice(negative_options)
                    
            elif objective == "Prevención":
                if rand_val < 0.5 and negative_options:
                    return random.choice(negative_options)
                elif rand_val < 0.8 and neutral_options:
                    return random.choice(neutral_options)
                elif positive_options:
                    return random.choice(positive_options)
                    
            elif objective == "Exploración":
                if rand_val < 0.5 and neutral_options:
                    return random.choice(neutral_options)
                elif rand_val < 0.75 and positive_options:
                    return random.choice(positive_options)
                elif negative_options:
                    return random.choice(negative_options)
                    
            elif objective == "Predeterminado":
                domain_weight = student_profile["ambitos"].get(chosen_domain, {}).get("peso", 50)
                
                if domain_weight < 40:
                    if rand_val < 0.6 and positive_options:
                        return random.choice(positive_options)
                    elif rand_val < 0.9 and neutral_options:
                        return random.choice(neutral_options)
                    elif negative_options:
                        return random.choice(negative_options)
                elif domain_weight > 70:
                    if rand_val < 0.4 and neutral_options:
                        return random.choice(neutral_options)
                    elif rand_val < 0.7 and negative_options:
                        return random.choice(negative_options)
                    elif positive_options:
                        return random.choice(positive_options)
                else:
                    if rand_val < 0.4 and neutral_options:
                        return random.choice(neutral_options)
                    elif rand_val < 0.7 and positive_options:
                        return random.choice(positive_options)
                    elif negative_options:
                        return random.choice(negative_options)
        
        options_by_preference = [positive_options, neutral_options, negative_options]
        for opt_group in options_by_preference:
            if opt_group:
                return random.choice(opt_group)
                
        return random.choice(matching_options)
    
    selected = random.choice(options)
    selected["ambito"] = chosen_domain
    return selected

def update_student_profile(student_profile, question, selected_option):
    try:
        q_ambitos = json.loads(question["ambitos"]) if isinstance(question["ambitos"], str) else q["ambitos"]
        q_ambitos = {str(k).strip(): v for k, v in q_ambitos.items() 
                    if k is not None and str(k).strip() and str(k).strip().lower() != "null"}
    except Exception as e:
        logger.warning(f"Error al procesar ámbitos de pregunta: {e}")
        q_ambitos = {selected_option.get("ambito", "General"): 100}
    
    option_peso = selected_option.get("peso", 0)
    
    emotional_profile = student_profile.get("perfil_emocional", "Neutro")
    if emotional_profile == "Positivo":
        option_peso *= random.uniform(0.9, 1.2)
    elif emotional_profile == "Negativo":
        option_peso *= random.uniform(0.8, 1.0)
    
    total_percentage = sum(valor for valor in q_ambitos.values())
    chosen_domain = selected_option.get("ambito")
    
    for domain, valor in q_ambitos.items():
        domain = str(domain).strip()
        if not domain or domain.lower() == "null":
            continue
            
        if domain not in student_profile["ambitos"]:
            student_profile["ambitos"][domain] = {
                "peso": 50,
                "porcentaje": 0,
                "count": 0,
                "history": []
            }
        
        factor = valor / total_percentage if total_percentage > 0 else 1
        
        if domain == chosen_domain:
            factor *= 1.5
        
        effect = option_peso * factor
        
        current_peso = student_profile["ambitos"][domain]["peso"]
        new_peso = current_peso + effect
        new_peso = max(0, min(new_peso, 100))
        student_profile["ambitos"][domain]["peso"] = new_peso
        
        if domain == chosen_domain:
            student_profile["ambitos"][domain]["count"] += 1.5
        else:
            student_profile["ambitos"][domain]["count"] += 0.5
        
        student_profile["ambitos"][domain]["history"].append(effect)
        if len(student_profile["ambitos"][domain]["history"]) > 10:
            student_profile["ambitos"][domain]["history"] = student_profile["ambitos"][domain]["history"][-10:]

def update_percentages(student_profile):
    total_count = sum(data.get("count", 0) for data in student_profile["ambitos"].values())
    
    if total_count > 0:
        for domain, data in student_profile["ambitos"].items():
            data["porcentaje"] = data.get("count", 0) / total_count
    else:
        for domain, data in student_profile["ambitos"].items():
            data["porcentaje"] = 0

def build_question_options():
    opciones_df = get_opciones_pregunta()
    question_options = {}
    
    for idx, row in opciones_df.iterrows():
        if isinstance(row.get("idPregunta"), dict) and "$oid" in row["idPregunta"]:
            q_id = row["idPregunta"]["$oid"]
        else:
            q_id = str(row.get("idPregunta", ""))
        
        if not q_id:
            continue
            
        if q_id not in question_options:
            question_options[q_id] = []
            
        option_dict = row.to_dict()
        question_options[q_id].append(option_dict)
    
    logger.info(f"Cargadas opciones para {len(question_options)} preguntas")
    return question_options

def simulate_responses(students, student_profiles, questions, question_options, responses_per_student):
    responses = []
    initial_domains_counter = {domain: 0 for domain in DOMAINS}
    
    for q_id, options in question_options.items():
        if isinstance(q_id, dict) and "$oid" in q_id:
            q_id = q_id["$oid"]
            
        for option in options:
            if isinstance(option.get("_id"), dict) and "$oid" in option["_id"]:
                option["_id"] = option["_id"]["$oid"]
    
    logger.info(f"Iniciando simulación de {responses_per_student} respuestas para {len(students)} alumnos")
    
    for student_idx, (student, student_profile) in enumerate(zip(students, student_profiles)):
        stu_id = student["_id"]["$oid"] if isinstance(student["_id"], dict) else str(student["_id"])
        
        # Trabajar con una copia del perfil interno para no modificar el original
        working_profile = {
            "ambitos": student_profile["ambitos"].copy() if isinstance(student_profile["ambitos"], dict) else 
                      json.loads(student_profile["ambitos"]),
            "objetivo": student_profile["objetivo"],
            "perfil_emocional": student_profile.get("perfil_emocional", "Neutro")
        }
        
        used_questions = set()
        student_domain_counter = Counter()
        
        for resp_idx in range(responses_per_student):
            is_first = (resp_idx == 0)
            chosen_domain = select_domain_for_objective(
                working_profile, 
                initial_domains_counter if is_first else None,
                is_first_question=is_first
            )
            
            student_domain_counter[chosen_domain] += 1
            
            question, used_questions = select_question(questions, chosen_domain, used_questions)
            
            selected_option = select_option_for_domain(question, question_options, chosen_domain, working_profile)
            if not selected_option:
                logger.warning(f"No se pudo seleccionar opción para pregunta {question.get('_id')}")
                continue
                
            update_student_profile(working_profile, question, selected_option)
            
            fecha = simulate_response_date(BASE_DATE, student_idx, resp_idx, len(students), responses_per_student)
            
            response = {
                "_id": {"$oid": generate_objectid()},
                "idAlumno": {"$oid": stu_id},
                "idPregunta": {"$oid": str(question["_id"]["$oid"]) if isinstance(question["_id"], dict) else str(question["_id"])},
                "idOpcionPregunta": {"$oid": str(selected_option["_id"]["$oid"]) if isinstance(selected_option["_id"], dict) else str(selected_option["_id"])},
                "fechaRespuesta": {"$date": fecha}
            }
            
            responses.append(response)
            update_percentages(working_profile)
        
        logger.debug(f"Alumno {student['nombre']} ({working_profile['objetivo']}, {working_profile['perfil_emocional']}): Distribución de dominios: {dict(student_domain_counter)}")
        
        # Actualizar el alumno real (sin perfil emocional)
        clean_ambitos = {}
        for domain, data in working_profile["ambitos"].items():
            clean_ambitos[domain] = {
                "peso": round(data["peso"], 1),
                "porcentaje": round(data["porcentaje"], 4)
            }
        student["ambitos"] = clean_ambitos
        
        if (student_idx + 1) % 10 == 0:
            logger.info(f"Generadas respuestas para {student_idx + 1}/{len(students)} alumnos")
    
    ambito_counts = Counter()
    for resp in responses:
        q_id = resp["idPregunta"]["$oid"] if isinstance(resp["idPregunta"], dict) else str(resp["idPregunta"])
        for q in questions:
            if str(q["_id"]["$oid"]) if isinstance(q["_id"], dict) else str(q["_id"]) == q_id:
                try:
                    q_ambitos = json.loads(q["ambitos"]) if isinstance(q["ambitos"], str) else q["ambitos"]
                    for ambito in q_ambitos.keys():
                        ambito_counts[ambito] += 1
                except:
                    continue
    
    logger.info(f"Distribución final de ámbitos en respuestas: {dict(ambito_counts)}")
    return responses

def analyze_generated_data(students, student_profiles, responses, questions):
    logger.info("Analizando datos generados")
    
    objective_counter = Counter([s["objetivo"] for s in students])
    emotional_counter = Counter([s.get("perfil_emocional", "Neutro") for s in student_profiles])
    
    logger.info(f"Distribución de objetivos: {dict(objective_counter)}")
    logger.info(f"Distribución de perfiles emocionales (interno): {dict(emotional_counter)}")
    
    ambito_weights_by_profile = {profile: {domain: [] for domain in DOMAINS} for profile in EMOTIONAL_PROFILES}
    
    for student_profile in student_profiles:
        ambitos = student_profile["ambitos"]
        profile = student_profile.get("perfil_emocional", "Neutro")
        
        for domain, data in ambitos.items():
            if domain in ambito_weights_by_profile[profile]:
                if isinstance(data, dict):
                    ambito_weights_by_profile[profile][domain].append(data.get("peso", 50))
                else:
                    ambito_weights_by_profile[profile][domain].append(data if isinstance(data, (int, float)) else 50)
    
    logger.info("Estadísticas de pesos por ámbito y perfil emocional:")
    for profile in EMOTIONAL_PROFILES:
        logger.info(f"=== Perfil {profile} ===")
        for domain in DOMAINS:
            weights = ambito_weights_by_profile[profile][domain]
            if weights:
                logger.info(f"{domain}: min={min(weights):.1f}, avg={sum(weights)/len(weights):.1f}, max={max(weights):.1f}")
    
    objective_domain_counts = {obj: {domain: 0 for domain in DOMAINS} for obj in OBJECTIVES}
    
    for resp in responses:
        stu_id = resp["idAlumno"]["$oid"] if isinstance(resp["idAlumno"], dict) else str(resp["idAlumno"])
        q_id = resp["idPregunta"]["$oid"] if isinstance(resp["idPregunta"], dict) else str(resp["idPregunta"])
        
        matching_students = [s for s in students if 
                            (s["_id"]["$oid"] if isinstance(s["_id"], dict) else str(s["_id"])) == stu_id]
        
        matching_questions = [q for q in questions if 
                             (q["_id"]["$oid"] if isinstance(q["_id"], dict) else str(q["_id"])) == q_id]
        
        if matching_students and matching_questions:
            objective = matching_students[0]["objetivo"]
            q = matching_questions[0]
            
            try:
                q_ambitos = json.loads(q["ambitos"]) if isinstance(q["ambitos"], str) else q["ambitos"]
                if q_ambitos:
                    main_domain = max(q_ambitos.items(), key=lambda x: x[1])[0]
                    objective_domain_counts[objective][main_domain] += 1
            except Exception as e:
                logger.warning(f"Error al analizar ámbitos de pregunta: {e}")
                continue
    
    logger.info("Distribución de ámbitos por objetivo del alumno:")
    for obj in OBJECTIVES:
        total = sum(objective_domain_counts[obj].values())
        if total > 0:
            percentages = {d: (count/total*100) for d, count in objective_domain_counts[obj].items()}
            sorted_percentages = sorted(percentages.items(), key=lambda x: x[1], reverse=True)
            logger.info(f"{obj}: {', '.join([f'{d}: {p:.1f}%' for d, p in sorted_percentages])}")
        else:
            logger.info(f"{obj}: No hay datos")

def main():
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
    
    file_handler = logging.FileHandler(os.path.join(OUTPUT_DIR, "generate_data.log"))
    file_handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
    logger.addHandler(file_handler)
    
    logger.info("Iniciando generación de datos sintéticos")
    
    # Generar alumnos y perfiles internos separados
    students, student_profiles = generate_students(N_STUDENTS)
    
    preguntas_df = get_preguntas()
    questions = preguntas_df.to_dict(orient="records")
    logger.info(f"Cargadas {len(questions)} preguntas de la base de datos")
    
    question_options = build_question_options()
    
    responses = simulate_responses(students, student_profiles, questions, question_options, RESPONSES_PER_STUDENT)
    
    analyze_generated_data(students, student_profiles, responses, questions)
    
    # Convertir ámbitos a formato JSON antes de guardar
    for student in students:
        if isinstance(student["ambitos"], dict):
            student["ambitos"] = json.dumps(student["ambitos"], ensure_ascii=False)
    
    students_file = os.path.join(OUTPUT_DIR, "alumnos.json")
    with open(students_file, "w", encoding="utf-8") as f:
        json.dump(students, f, ensure_ascii=False, indent=2)
    logger.info(f"Se han generado {len(students)} alumnos en '{students_file}'")
    
    responses_file = os.path.join(OUTPUT_DIR, "respuestas.json")
    with open(responses_file, "w", encoding="utf-8") as f:
        json.dump(responses, f, ensure_ascii=False, indent=2)
    logger.info(f"Se han generado {len(responses)} respuestas en '{responses_file}'")
    
    logger.info("Generación de datos completada con éxito")

if __name__ == "__main__":
    main()