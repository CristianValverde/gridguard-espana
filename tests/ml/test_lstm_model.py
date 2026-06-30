"""
Tests for ml.train.lstm_model.LSTMAutoencoderModule.

Verifies that:
1. Forward pass produces the correct output shape.
2. torch.save / torch.load round-trip works (the serialization bug we fixed).
3. The loaded model's __class__.__module__ is 'ml.train.lstm_model' — the
   exact invariant that was broken when the class lived inside __new__.
"""
import pytest

torch = pytest.importorskip("torch")


def test_forward_shape():
    from ml.train.lstm_model import LSTMAutoencoderModule

    model = LSTMAutoencoderModule(n_features=4, hidden_size=8, num_layers=1, dropout=0.0)
    x = torch.randn(2, 5, 4)
    out = model(x)
    assert out.shape == (2, 5, 4), f"Expected (2, 5, 4) got {tuple(out.shape)}"


def test_serialization_roundtrip(tmp_path):
    """Core regression: torch.save must not raise PicklingError."""
    from ml.train.lstm_model import LSTMAutoencoderModule

    model = LSTMAutoencoderModule(n_features=4, hidden_size=8, num_layers=1, dropout=0.0)

    path = tmp_path / "m.pth"
    torch.save({"model": model}, str(path))

    loaded = torch.load(str(path), weights_only=False)
    loaded_model = loaded["model"]

    # The class must be importable from its module (not a local function scope)
    assert loaded_model.__class__.__module__ == "ml.train.lstm_model", (
        f"Wrong module: {loaded_model.__class__.__module__!r} — "
        "pickle cannot reconstruct local classes"
    )

    # Forward pass on the reloaded model must work
    x = torch.randn(2, 5, 4)
    out = loaded_model(x)
    assert out.shape == (2, 5, 4), f"Expected (2, 5, 4) got {tuple(out.shape)}"
