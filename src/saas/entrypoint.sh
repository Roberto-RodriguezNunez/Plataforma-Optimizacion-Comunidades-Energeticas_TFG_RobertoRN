#!/bin/sh
# Esperar a que PostgreSQL esté listo
echo "⏳ Esperando PostgreSQL..."
until python3 -c "import os, psycopg2; psycopg2.connect(os.environ['DATABASE_URL']).close()" 2>/dev/null; do
  sleep 1
done
echo "✅ PostgreSQL listo."

# Cargar seed si la BD está vacía (idempotente)
python3 seed.py

# Arrancar Flask
exec python3 run.py
