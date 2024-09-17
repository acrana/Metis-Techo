import sqlite3
import pandas as pd

# Connect to the SQLite database
conn = sqlite3.connect('clinical_decision_support.db')

# Read tables into pandas DataFrames
demographics_df = pd.read_sql_query("SELECT * FROM TBL_Demographics", conn)
survey_df = pd.read_sql_query("SELECT * FROM TBL_Survey", conn)
ade_records_df = pd.read_sql_query("SELECT * FROM TBL_ADERecords", conn)

# Close the connection
conn.close()
