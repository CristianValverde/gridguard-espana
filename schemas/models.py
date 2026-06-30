"""
GridGuard España — Pydantic data models.

Única fuente de verdad para todos los schemas del sistema.
Importar desde aquí en tools, mcp_servers, agents y eval.
"""

from __future__ import annotations

from datetime import datetime
from typing import Literal, Optional

from pydantic import BaseModel, Field, field_validator


# ---------------------------------------------------------------------------
# Capa 1 — Modelos de datos de red eléctrica
# ---------------------------------------------------------------------------

class GridSnapshot(BaseModel):
    """Snapshot normalizado de la red eléctrica. Salida de data_normalizer()."""

    timestamp: datetime
    region: str = "peninsular"
    demand_mw: float = Field(..., gt=0, description="Demanda real en MW")
    generation_by_tech: dict[str, float] = Field(
        ...,
        description=(
            "Generación por tecnología en MW. "
            "Keys: solar, wind, nuclear, hydro, gas, coal, cogeneration, other"
        ),
    )
    interchange_flows: dict[str, float] = Field(
        ...,
        description="Flujos de intercambio en MW por frontera. Positivo=exportación.",
    )
    reserve_mw: float = Field(..., description="Reserva de regulación secundaria en MW")
    programmed_generation_mw: Optional[float] = None
    source_timestamp: Optional[datetime] = Field(
        None, description="Timestamp original devuelto por ESIOS"
    )


class StabilityProxies(BaseModel):
    """
    Proxies de estabilidad de frecuencia calculados desde GridSnapshot.
    Salida de stability_proxy_calculator().
    La frecuencia de red real (Hz) no está disponible públicamente vía API.
    """

    balance_mw: float = Field(
        ..., description="Generación total − Demanda real en MW"
    )
    reserve_activated_pct: float = Field(
        ..., ge=0.0, le=100.0, description="% de reserva de regulación activada"
    )
    interchange_delta_mw: float = Field(
        ..., description="Variación brusca en flujos de intercambio en MW"
    )
    renewables_pct: float = Field(
        ..., ge=0.0, le=100.0,
        description="(Solar + Eólica) / Demanda total × 100",
    )
    composite_pressure: float = Field(
        ..., ge=0.0, le=1.0,
        description="Indicador compuesto de presión sobre frecuencia [0-1]",
    )


class Baseline(BaseModel):
    """Estadísticas históricas de referencia. Salida de get_baseline() del MCP memory."""

    demand_mean_mw: float
    demand_std_mw: float
    renewables_pct_mean: float
    renewables_pct_std: float
    balance_mean_mw: float
    balance_std_mw: float
    window_hours: int = Field(..., gt=0)
    computed_at: datetime


class RiskScore(BaseModel):
    """Score de riesgo combinado. Salida de risk_scorer()."""

    value: int = Field(..., ge=0, le=100, description="Score de riesgo [0-100]")
    components: dict[str, float] = Field(
        ...,
        description="Contribución de cada componente: ml_score, balance_pressure, renewables_risk",
    )
    level: Literal["LOW", "MEDIUM", "HIGH", "CRITICAL"]
    blackout_probability: float = Field(..., ge=0.0, le=1.0)
    timestamp: datetime = Field(default_factory=datetime.utcnow)

    @field_validator("level", mode="before")
    @classmethod
    def derive_level(cls, v: str) -> str:
        return v.upper()


class Alert(BaseModel):
    """Alerta estructurada. Salida de alert_formatter()."""

    severity: Literal["LOW", "MEDIUM", "HIGH", "CRITICAL"]
    scope: Literal["nacional", "peninsular", "zonal", "local"]
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    summary: str
    details: str
    recommended_action: str


# ---------------------------------------------------------------------------
# Capa 2 — MCP: ree-grid (ESIOS API)
# ---------------------------------------------------------------------------

class DemandSnapshot(BaseModel):
    """Demanda en tiempo real. Salida de get_realtime_demand()."""

    region: str
    timestamp: datetime
    demand_mw: float = Field(..., gt=0)
    programmed_mw: Optional[float] = None


class GenerationMix(BaseModel):
    """Mix de generación en un intervalo. Salida de get_generation_mix()."""

    start: datetime
    end: datetime
    technology_mw: dict[str, float] = Field(
        ..., description="MW por tecnología (solar, wind, nuclear, hydro, gas, …)"
    )
    total_mw: float = Field(..., gt=0)


class InterchangeFlow(BaseModel):
    """Flujo de intercambio en una frontera. Salida de get_interchange_flows()."""

    border: str = Field(..., description="Frontera: 'ES-FR', 'ES-PT', 'ES-MA'")
    timestamp: datetime
    flow_mw: float = Field(
        ..., description="Positivo = exportación desde España"
    )
    capacity_mw: Optional[float] = Field(None, description="Capacidad nominal de la interconexión")


class ProgrammedGeneration(BaseModel):
    """Generación programada. Salida de get_programmed_generation()."""

    timestamp: datetime
    scheduled_mw: float = Field(..., gt=0)
    available_units: list[str] = Field(default_factory=list)


class ReserveStatus(BaseModel):
    """Estado de las reservas de regulación. Salida de get_reserve_activation()."""

    timestamp: datetime
    secondary_reserve_mw: float = Field(..., ge=0)
    tertiary_reserve_mw: float = Field(..., ge=0)
    activated_secondary_pct: float = Field(..., ge=0.0, le=100.0)
    activated_tertiary_pct: float = Field(..., ge=0.0, le=100.0)


# ---------------------------------------------------------------------------
# Capa 2 — MCP: weather (AEMET + OpenMeteo)
# ---------------------------------------------------------------------------

class WindForecast(BaseModel):
    """Previsión de viento por zona. Salida de get_wind_forecast()."""

    zone: str
    forecast_hours: int = Field(..., gt=0)
    timestamps: list[datetime]
    wind_speed_ms: list[float] = Field(..., description="Velocidad del viento en m/s")
    generation_capacity_pct: list[float] = Field(
        ..., description="% estimado de capacidad eólica instalada"
    )


class RadiationForecast(BaseModel):
    """Previsión de radiación solar por zona. Salida de get_solar_radiation()."""

    zone: str
    forecast_hours: int = Field(..., gt=0)
    timestamps: list[datetime]
    radiation_wm2: list[float] = Field(..., description="Radiación en W/m²")
    generation_capacity_pct: list[float] = Field(
        ..., description="% estimado de capacidad solar instalada"
    )


class MeteoAlert(BaseModel):
    """Alerta meteorológica de AEMET. Salida de get_aemet_alerts()."""

    alert_id: str
    zone: str
    phenomenon: str = Field(..., description="WIND, STORM, HEAT, RAIN, SNOW, …")
    severity: Literal["YELLOW", "ORANGE", "RED"]
    start: datetime
    end: datetime
    description: str


# ---------------------------------------------------------------------------
# Capa 2 — MCP: entso-e
# ---------------------------------------------------------------------------

class CrossBorderFlow(BaseModel):
    """Flujo transfronterizo europeo. Salida de get_cross_border_flows()."""

    from_country: str
    to_country: str
    timestamp: datetime
    flow_mw: float
    interconnection_capacity_mw: Optional[float] = None


class EUGenerationSnapshot(BaseModel):
    """Snapshot de generación en un país europeo. Salida de get_eu_generation()."""

    country: str
    timestamp: datetime
    technology_mw: dict[str, float]
    total_mw: float = Field(..., gt=0)


class CapacityMargin(BaseModel):
    """Margen de capacidad disponible. Salida de get_capacity_margin()."""

    country: str
    timestamp: datetime
    available_capacity_mw: float = Field(..., ge=0)
    peak_demand_mw: float = Field(..., gt=0)
    margin_pct: float = Field(..., description="(available - peak) / peak × 100")


class MarketPrice(BaseModel):
    """Precio day-ahead por país. Salida de get_eu_market_prices().

    Resolución real ENTSO-E (A44): 15-min (96 valores/día), NO horaria.
    Confirmado via api-probe 2026-06-30. Índice convertido a UTC.
    """

    country: str
    timestamps: list[datetime]
    price_eur_mwh: list[float]


# ---------------------------------------------------------------------------
# Capa 2 — MCP: memory (SQLite)
# ---------------------------------------------------------------------------

class Incident(BaseModel):
    """Incidente registrado en la base de datos local. Devuelto por query_incidents()."""

    incident_id: str
    start: datetime
    end: Optional[datetime] = None
    severity: Literal["LOW", "MEDIUM", "HIGH", "CRITICAL"]
    description: str
    resolution: Optional[str] = None
    risk_score_at_onset: int = Field(..., ge=0, le=100)


# ---------------------------------------------------------------------------
# Capa 2 — MCP: ml-models (inferencia determinística)
# ---------------------------------------------------------------------------

class AnomalyResult(BaseModel):
    """Resultado del modelo de detección de anomalías. Salida de get_anomaly_score()."""

    anomaly_score: float = Field(..., ge=0.0, le=1.0)
    is_anomaly: bool
    top_features: list[str] = Field(
        ..., description="Features con mayor contribución a la anomalía"
    )
    feature_scores: dict[str, float]
    model_type: Literal["isolation_forest", "lstm_ae"]
    inference_time_ms: float = Field(..., ge=0)


class LoadForecast(BaseModel):
    """Previsión de demanda. Salida de get_load_forecast()."""

    horizon_hours: int = Field(..., gt=0)
    timestamps: list[datetime]
    demand_mw: list[float]
    lower_bound_mw: list[float]
    upper_bound_mw: list[float]
    confidence_pct: float = Field(
        ..., ge=0.0, le=100.0, description="Intervalo de confianza, ej. 95.0"
    )
    model_type: Literal["xgboost", "prophet"]


class BlackoutRisk(BaseModel):
    """Probabilidad de apagón calibrada. Salida de get_blackout_probability()."""

    probability: float = Field(..., ge=0.0, le=1.0, description="P(blackout_1h)")
    risk_level: Literal["LOW", "MEDIUM", "HIGH", "CRITICAL"]
    feature_importance: dict[str, float]
    calibration_method: str = Field(default="isotonic_regression")

    @field_validator("risk_level", mode="before")
    @classmethod
    def infer_level_from_probability(cls, v: str) -> str:
        return v.upper()


# ---------------------------------------------------------------------------
# Capa 2 — MCP: knowledge-base (ChromaDB RAG)
# ---------------------------------------------------------------------------

class RegulatoryChunk(BaseModel):
    """Fragmento de normativa técnica. Salida de search_regulations()."""

    chunk_id: str
    document: str = Field(..., description="Ej: 'PO_1.1', 'EN_50160', 'ENTSO-E_NC_OS'")
    section: Optional[str] = None
    content: str
    relevance_score: float = Field(..., ge=0.0, le=1.0)


class IncidentRecord(BaseModel):
    """Registro de post-mortem histórico. Salida de find_similar_incident()."""

    incident_id: str
    date: datetime
    title: str
    description: str
    cause: str
    outcome: str
    similarity_score: float = Field(..., ge=0.0, le=1.0)


class Procedure(BaseModel):
    """Procedimiento operacional de REE. Salida de get_mitigation_procedure()."""

    procedure_id: str
    title: str
    situation: str
    steps: list[str]
    authority: str = Field(..., description="REE, ENTSO-E, …")


# ---------------------------------------------------------------------------
# Capa 3 — Skills (AgentTool outputs)
# ---------------------------------------------------------------------------

class Anomaly(BaseModel):
    """Anomalía individual detectada por AnomalyDetectorSkill."""

    feature: str
    value: float
    expected_range: tuple[float, float] = Field(
        ..., description="(min_normal, max_normal)"
    )
    deviation_sigma: float = Field(
        ..., description="Desviaciones estándar respecto al baseline"
    )
    description: str


class AnomalyReport(BaseModel):
    """Output estructurado de AnomalyDetectorSkill."""

    anomalies: list[Anomaly]
    severity: Literal["LOW", "MEDIUM", "HIGH", "CRITICAL"]
    confidence: float = Field(..., ge=0.0, le=1.0)
    ml_score: float = Field(..., ge=0.0, le=1.0, description="Score del modelo ML [0-1]")
    regulatory_context: str = Field(
        ..., description="Fragmento normativo relevante recuperado del RAG"
    )
    reasoning: str = Field(..., description="Razonamiento del LLM sobre la situación")
    timestamp: datetime = Field(default_factory=datetime.utcnow)


class RenewablesForecast(BaseModel):
    """Output estructurado de RenewablesForecastSkill."""

    risk_level: Literal["LOW", "MEDIUM", "HIGH", "CRITICAL"]
    expected_generation_pct: float = Field(
        ..., ge=0.0, le=100.0,
        description="% previsto de renovables sobre demanda en ventana crítica",
    )
    critical_window: str = Field(
        ..., description="Ventana horaria de mayor riesgo, ej. '14:00-18:00 UTC'"
    )
    ml_forecast_mw: float = Field(..., ge=0, description="Previsión ML de demanda en MW")
    reasoning: str
    timestamp: datetime = Field(default_factory=datetime.utcnow)


class ComplianceReport(BaseModel):
    """Output estructurado de RegulatoryComplianceSkill."""

    applicable_regulations: list[dict] = Field(
        default_factory=list,
        description="[{id, section, trigger}] de procedimientos REE/ENTSO-E aplicables",
    )
    mandatory_actions: list[str] = Field(default_factory=list)
    compliance_status: Literal["compliant", "warning", "violation"]
    reasoning: str
    timestamp: datetime = Field(default_factory=datetime.utcnow)


# ---------------------------------------------------------------------------
# Capa 4 — Sub-agent outputs
# ---------------------------------------------------------------------------

class CrossBorderAnalysis(BaseModel):
    """Output de CrossBorderAgent."""

    can_import: bool
    available_capacity_mw: float = Field(..., ge=0)
    countries: list[str] = Field(..., description="Países con capacidad disponible")
    risk_mitigation: str = Field(
        ..., description="Procedimiento de mitigación recomendado"
    )
    timestamp: datetime = Field(default_factory=datetime.utcnow)


# ---------------------------------------------------------------------------
# Cross-cutting — Guardrails / Hooks
# ---------------------------------------------------------------------------

class AgentOutput(BaseModel):
    """
    Wrapper de salida usado por after_agent_callback para validar bounds
    antes de que el output llegue a capas superiores.
    """

    agent_name: str
    risk_score: Optional[int] = Field(None, ge=0, le=100)
    blackout_probability: Optional[float] = Field(None, ge=0.0, le=1.0)
    severity: Optional[Literal["LOW", "MEDIUM", "HIGH", "CRITICAL"]] = None
    scope: Optional[Literal["nacional", "peninsular", "zonal", "local"]] = None
    output_data: dict
    timestamp: datetime = Field(default_factory=datetime.utcnow)
