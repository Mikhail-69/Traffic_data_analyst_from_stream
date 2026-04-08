from airflow import DAG
from airflow.operators.bash import BashOperator
from airflow.operators.python import PythonOperator
from datetime import datetime, timedelta
import sqlite3
import os

# Пути к твоим скриптам (внутри контейнера Airflow)
# Если твои файлы лежат на хосте в C:\champion_project2,
# они должны быть смонтированы в контейнер. В docker-compose из задания
# папка ./data монтируется в /opt/airflow/data. Предположим, ты смонтируешь
# весь проект в /opt/airflow/project. Но для простоты используем BashOperator
# с вызовом python /opt/airflow/data/run_full_pipeline.py (если скрипт там).
# Уточни путь по факту.

PROJECT_PATH = "/opt/airflow/data"   # или куда смонтировал

default_args = {
    'owner': 'champion',
    'depends_on_past': False,
    'start_date': datetime(2026, 4, 1),
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
}

# Функция очистки и нормализации (выполняется после детекции)
def clean_and_normalize():
    db_path = os.path.join(PROJECT_PATH, "traffic.db")
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    # 1. Удалить дубликаты (оставить первый по id)
    cur.execute("""
        DELETE FROM detections 
        WHERE id NOT IN (SELECT MIN(id) FROM detections GROUP BY frame_id, timestamp)
    """)
    # 2. Привести vehicle_type к нижнему регистру
    cur.execute("UPDATE detections SET vehicle_type = LOWER(vehicle_type)")
    # 3. Удалить записи с низкой уверенностью (<0.3)
    cur.execute("DELETE FROM detections WHERE confidence < 0.3")
    conn.commit()
    conn.close()
    print("Очистка и нормализация завершены")

# Функция обогащения (добавить час, день недели, можно погоду)
def enrich():
    db_path = os.path.join(PROJECT_PATH, "traffic.db")
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    try:
        cur.execute("ALTER TABLE detections ADD COLUMN hour INTEGER")
    except:
        pass
    try:
        cur.execute("ALTER TABLE detections ADD COLUMN dayofweek INTEGER")
    except:
        pass
    cur.execute("SELECT id, timestamp FROM detections")
    rows = cur.fetchall()
    for row_id, ts_str in rows:
        # ts_str имеет формат '2024-01-01 12:00:00.123'
        dt = datetime.strptime(ts_str, '%Y-%m-%d %H:%M:%S.%f')
        cur.execute("UPDATE detections SET hour = ?, dayofweek = ? WHERE id = ?",
                    (dt.hour, dt.weekday(), row_id))
    conn.commit()
    conn.close()
    print("Обогащение завершено")

# DAG определение
with DAG(
    dag_id='traffic_monitoring_etl',
    default_args=default_args,
    schedule_interval='*/15 * * * *',   # каждые 15 минут
    catchup=False,                      # не выполнять за пропущенные интервалы
    tags=['traffic', 'etl']
) as dag:

    # Задача 1: запуск полного пайплайна (детекция + метрики)
    run_detection = BashOperator(
        task_id='run_full_pipeline',
        bash_command=f'python {PROJECT_PATH}/run_full_pipeline.py',
    )

    # Задача 2: очистка и нормализация
    clean_task = PythonOperator(
        task_id='clean_and_normalize',
        python_callable=clean_and_normalize,
    )

    # Задача 3: обогащение
    enrich_task = PythonOperator(
        task_id='enrich',
        python_callable=enrich,
    )

    # Задача 4: переобучение модели (можно запускать раз в сутки, но для демо – после обогащения)
    retrain_model = BashOperator(
        task_id='retrain_model',
        bash_command=f'python {PROJECT_PATH}/train_model.py',
    )

    run_detection >> clean_task >> enrich_task >> retrain_model