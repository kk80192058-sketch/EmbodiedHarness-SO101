"""Read-only pairing of SO-101 session events with their image observations.

Each bounded step writes a ``write_intent``, emits exactly one ``result``, and
surrounds that interval with an observation.  Pairing by image filename or by
the first event in a session is unsafe when a persistent session has multiple
steps, so this module preserves the event order explicitly.
"""
from __future__ import annotations

from typing import Any


def pair_completed_steps(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Return completed steps with their immediately surrounding observations.

    An incomplete interval is retained nowhere: it cannot supply a visual
    response sample because there is no evidence tying its before/after frames
    to one specific motor write.  The caller may separately report this count.
    """
    paired: list[dict[str, Any]] = []
    pending: dict[str, Any] | None = None
    before: dict[str, Any] | None = None
    awaiting_after: dict[str, Any] | None = None

    for row in rows:
        kind = row.get('event')
        payload = row.get('payload')
        if not isinstance(payload, dict):
            continue
        if kind == 'observation':
            if awaiting_after is not None:
                awaiting_after['after'] = payload
                paired.append(awaiting_after)
                awaiting_after = None
            else:
                before = payload
        elif kind == 'write_intent':
            # A new intent before a result makes the old interval ambiguous.
            pending = {'intent': payload, 'before': before}
        elif kind == 'result' and pending is not None:
            awaiting_after = {**pending, 'result': payload}
            pending = None
    return [entry for entry in paired if entry['before'] is not None]
