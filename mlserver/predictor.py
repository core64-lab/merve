"""Predictor contract for user-supplied model classes (RFC 0001, D13).

A predictor is any Python class whose instances expose ``predict(X)``. The
:class:`Predictor` protocol below formalizes that contract structurally, so
user classes never need to import or inherit anything from this package::

    class MyPredictor:                  # satisfies Predictor - no import needed
        def predict(self, X):
            return self.model.predict(X)

Optional methods (discovered via ``hasattr`` at runtime, intentionally NOT
part of the protocol body so ``isinstance`` checks only require ``predict``):

``predict_proba(X)``
    Return per-class probabilities. When present, the ``/predict_proba``
    endpoint serves it; otherwise that endpoint responds 501.

``load() -> None``
    Called exactly once at server startup, after ``__init__`` and before the
    first prediction (including warmup). Put expensive artifact loading here
    (model weights, tokenizers, ...) so construction stays cheap and startup
    failures surface as clear errors instead of failing the first request.
    If ``load()`` raises, the server does not start and the predictor is
    never marked ready.

``close() -> None``
    Called at server shutdown for resource cleanup.
"""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable


@runtime_checkable
class Predictor(Protocol):
    """Structural interface for classes served by mlserver.

    Only ``predict`` is required. ``isinstance(obj, Predictor)`` checks for
    the presence of a callable ``predict`` attribute (runtime protocols check
    method presence, not signatures). See the module docstring for the
    optional ``predict_proba``/``load``/``close`` hooks.
    """

    def predict(self, X: Any) -> Any:
        """Return predictions for a 2D feature matrix ``X``."""
        ...
