Fase 1: Cimientos (empezar aquí)
  - schemas/models.py (Pydantic: GridSnapshot, RiskScore, AnomalyReport...)
  - config/model_loader.py + config/model_registry.yaml
  - data/download_esios.py, download_aemet.py, download_entsoe.py
  - data/build_dataset.py

Fase 2: MCP Servers
  - mcp/ree_grid/ — wraper ESIOS
  - mcp/ml_models/ — carga pesos y sirve inferencia
  (uno por sesión de Claude Code)

Fase 3: ML Training
  - ml/train/train_anomaly.py (LSTM Autoencoder + CUDA)
  - ml/train/train_forecast.py (XGBoost GPU)
  - ml/train/train_classifier.py (GBM calibrado)
  - ml/retrain_pipeline.py (cron mensual)

Fase 4: Skills y Sub-agentes
  - agents/skills/anomaly_detector.py
  - agents/skills/renewables_forecast.py
  - agents/sub/report_agent.py
  - agents/sub/cross_border_agent.py

Fase 5: Orchestrator + Hooks + Eval
  - agents/orchestrator.py
  - guardrails/hooks.py
  - eval/test_blackout_apr2025.py
