import psycopg2
from sqlalchemy import create_engine

def get_db_connection():
    """Create PostgreSQL database connection"""
    try:
        connection = psycopg2.connect(
            database="mimic",
            user="your_username",
            password="your_password",
            host="localhost",
            port="5432"
        )
        return connection
    except Exception as e:
        print(f"Error connecting to database: {e}")
        return None

def get_sqlalchemy_engine():
    """Create SQLAlchemy engine for pandas operations"""
    return create_engine('postgresql://your_username:your_password@localhost:5432/mimic')