# GridGuard España — Instrucciones para Claude Code

## Arquitectura completa
Ver ARCHITECTURE.md

## Restricciones NO NEGOCIABLES
- NUNCA leer .env, *.key, secrets.*, credentials.* ni ningún fichero de credenciales
- Todos los tokens (ESIOS_TOKEN, AEMET_API_KEY, etc.) vienen de variables de entorno
- NUNCA hardcodear nombres de modelos Gemini — siempre usar config/model_loader.py
- Principio shift-left: Python puro > ML > LLM. No usar LLM donde una función basta.

## Orden de implementación
Ver ROADMAP.md
