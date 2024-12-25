from google.cloud import bigquery
from functools import lru_cache
from config import PROJECT_ID

def get_mimic_connection():
    return bigquery.Client(project=PROJECT_ID)

def execute_query(query):
    client = get_mimic_connection()
    try:
        df = client.query(query).to_dataframe()
        return df
    except Exception as e:
        print(f"Query failed: {e}")
        return None

@lru_cache(maxsize=128)
def cached_query(query_string):
    return execute_query(query_string)