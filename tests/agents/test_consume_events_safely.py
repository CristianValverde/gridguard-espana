"""_consume_events_safely: no dejar que una excepcion a mitad del generador
de Events del Runner aborte el ciclo completo sin producir resultado.

Spec: docs/superpowers/specs/2026-07-04-orchestrator-crash-resilience-design.md
"""
from __future__ import annotations

from agents.orchestrator import _consume_events_safely


def test_generador_normal_sin_excepcion():
    def _gen():
        yield "a"
        yield "b"
        yield "c"

    assert _consume_events_safely(_gen()) == ["a", "b", "c"]


def test_generador_lanza_a_mitad_conserva_lo_ya_capturado():
    def _gen():
        yield "a"
        yield "b"
        raise ValueError("simulacion de ValidationError de ADK")

    assert _consume_events_safely(_gen()) == ["a", "b"]


def test_generador_lanza_inmediatamente_devuelve_lista_vacia():
    def _gen():
        raise ValueError("simulacion de fallo inmediato")
        yield  # pragma: no cover — nunca se alcanza, hace de esto un generador

    assert _consume_events_safely(_gen()) == []


def test_generador_vacio():
    def _gen():
        return
        yield  # pragma: no cover

    assert _consume_events_safely(_gen()) == []
