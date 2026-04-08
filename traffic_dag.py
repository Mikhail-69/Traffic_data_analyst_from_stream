from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.operators.bash import BashOperator
from datetime import datetime, timedelta
import sqlite3
import pandas as pd
import os

# Пути внутри контейнера Airflow (папка /opt/airflow/data монтируется из вашей C:\champion_project2\data)
DB_PATH = '/opt/airflow/data/traffic.db'
VIDEO_PATH = '/opt/airflow/data/traffic_video.mp4'

# --- Функции, которые будут вызываться в задачах ---

def run_detection_incremental():
    # Проверяем, есть ли уже записи
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("SELECT COUNT(*) FROM detections")
    count = cursor.fetchone()[0]
    conn.close()
    
    if count == 0:
        import subprocess
        subprocess.run(['python', '/opt/airflow/data/run_full_pipeline.py'], check=True)
    else:
        print("Данные уже есть, пропускаем детекцию. Для инкремента нужно доработать.")

def clean_detections():
    """Удаление дубликатов, low confidence, нормализация типов."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("""
        DELETE FROM detections 
        WHERE id NOT IN (
            SELECT MIN(id) FROM detections 
            GROUP BY frame_id, timestamp
        )
    """)
    cursor.execute("UPDATE detections SET vehicle_type = LOWER(vehicle_type)")
    cursor.execute("DELETE FROM detections WHERE confidence < 0.3") 
    conn.commit()
    conn.close()
    print("Очистка detections выполнена.")

def enrich_detections():
    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql_query("SELECT id, timestamp FROM detections", conn)
    if df.empty:
        conn.close()
        return
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df['hour'] = df['timestamp'].dt.hour
    df['dayofweek'] = df['timestamp'].dt.dayofweek
    min_time = df['timestamp'].min()
    df['minute_num'] = (df['timestamp'] - min_time).dt.total_seconds() / 60
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS enriched_detections (
            id INTEGER PRIMARY KEY,
            timestamp TEXT,
            vehicle_type TEXT,
            confidence REAL,
            x REAL, y REAL, w REAL, h REAL,
            frame_id INTEGER,
            hour INTEGER,
            dayofweek INTEGER,
            minute_num REAL,
            weather TEXT
        )
    """)
   
    cursor.execute("DELETE FROM enriched_detections")
    
    # Добавим погоду-заглушку ('sunny')
    for _, row in df.iterrows():
        cursor.execute("""
            INSERT INTO enriched_detections 
            (id, timestamp, vehicle_type, confidence, x, y, w, h, frame_id, hour, dayofweek, minute_num, weather)
            SELECT id, timestamp, vehicle_type, confidence, x, y, w, h, frame_id, ?, ?, ?, 'sunny'
            FROM detections WHERE id = ?
        """, (row['hour'], row['dayofweek'], row['minute_num'], row['id']))
    conn.commit()
    conn.close()
    print("Обогащение завершено.")

def aggregate_metrics():
    """Пересчёт метрик из enriched_detections (аналогично calculate_metrics, но на обогащённых данных)."""
    conn = sqlite3.connect(DB_PATH)
    # Читаем из enriched_detections
    df = pd.read_sql_query("SELECT timestamp, vehicle_type, confidence FROM enriched_detections", conn)
    if df.empty:
        conn.close()
        return
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df['minute_bucket'] = df['timestamp'].dt.floor('min')
    metrics = df.groupby('minute_bucket').agg(
        total_vehicles=('vehicle_type', 'count'),
        avg_confidence=('confidence', 'mean')
    ).reset_index()
    
    cursor = conn.cursor()
    cursor.execute("DELETE FROM metrics")
    for _, row in metrics.iterrows():
        cursor.execute("""
            INSERT INTO metrics (minute_bucket, total_vehicles, avg_confidence)
            VALUES (?, ?, ?)
        """, (row['minute_bucket'], row['total_vehicles'], row['avg_confidence']))
    conn.commit()
    conn.close()
    print(f"Агрегировано {len(metrics)} минутных бакетов.")

def train_model_task():
    """Вызов train_model.py."""
    import subprocess
    subprocess.run(['python', '/opt/airflow/data/train_model.py'], check=True)

# --- Определение DAG ---
default_args = {
    'owner': 'champion',
    'depends_on_past': False,
    'start_date': datetime(2026, 4, 1),
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
}

dag = DAG(
    'traffic_etl_pipeline',
    default_args=default_args,
    description='ETL для транспортных данных: детекция, очистка, обогащение, агрегация, обучение модели',
    schedule_interval='*/15 * * * *',   # каждые 15 минут
    catchup=False,
    tags=['traffic', 'etl'],
)

# Задачи
t_detect = PythonOperator(
    task_id='run_detection',
    python_callable=run_detection_incremental,
    dag=dag,
)

t_clean = PythonOperator(
    task_id='clean_detections',
    python_callable=clean_detections,
    dag=dag,
)

t_enrich = PythonOperator(
    task_id='enrich_detections',
    python_callable=enrich_detections,
    dag=dag,
)

t_aggregate = PythonOperator(
    task_id='aggregate_metrics',
    python_callable=aggregate_metrics,
    dag=dag,
)

t_train = PythonOperator(
    task_id='train_model',
    python_callable=train_model_task,
    dag=dag,
)

# Определяем порядок (DAG)
t_detect >> t_clean >> t_enrich >> t_aggregate >> t_train