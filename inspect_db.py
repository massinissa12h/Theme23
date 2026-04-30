"""
Script temporaire pour inspecter les tables Supabase.
"""

from supabase import create_client, Client
from dotenv import load_dotenv
import os

load_dotenv()

url = os.environ.get("SUPABASE_URL")
key = os.environ.get("SUPABASE_KEY")

if not url or not key:
    raise RuntimeError("SUPABASE_URL et SUPABASE_KEY doivent être définis.")

client: Client = create_client(url, key)

print("--- Aperçu des tables ---")

tables = ["products", "interactions", "recommendations", "product_tags"]

for table_name in tables:
    print(f"\n>>> Table: {table_name}")
    try:
        resp = client.table(table_name).select("*").limit(10).execute()
        data = resp.data
        if data:
            for i, row in enumerate(data):
                print(f"  [{i}] {row}")
        else:
            print("  (vide)")
    except Exception as e:
        print(f"  Erreur: {e}")

print("\n--- Fin ---")