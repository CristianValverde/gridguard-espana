# tests/agents/test_regulatory_compliance.py
from schemas.models import ComplianceReport


def test_compliance_report_schema_valid():
    r = ComplianceReport(
        applicable_regulations=[{"id": "PO 7.4", "section": "§3.2", "trigger": "renovables >75%"}],
        mandatory_actions=["verificar reserva primaria"],
        compliance_status="warning",
        reasoning="alta penetración renovable",
    )
    assert r.compliance_status == "warning"


def test_compliance_status_rejects_invalid():
    import pytest
    from pydantic import ValidationError
    with pytest.raises(ValidationError):
        ComplianceReport(applicable_regulations=[], mandatory_actions=[],
                         compliance_status="invalido", reasoning="x")


def test_skill_module_imports():
    import agents.regulatory_compliance as rc
    assert rc.RegulatoryComplianceSkill is not None
    assert "search_regulations" in rc._INSTRUCTION
    assert "NUNCA" in rc._INSTRUCTION or "Only cite" in rc._INSTRUCTION  # no inventar procedimientos
