#!/usr/bin/env bash
# Executa todo o pipeline. Uso: ./rodar_tudo.sh  (requer .venv com requirements.txt)
set -e
cd "$(dirname "$0")/src"
PY=../.venv/bin/python
for f in 01_preparacao.py 02_exploratoria.py 03_q1_previsao.py 04_q2_mercados.py 05_q3_produtos.py; do
  echo ">>> $f"; $PY -W ignore "$f"
done
