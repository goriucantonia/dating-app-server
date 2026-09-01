"""The AI Interaction Module — the foundational service (`ai_interaction.md`).

Every AI call in this system has exactly one road to travel:

    task name -> TaskRouter -> AIProvider -> Structured Output Guard
              -> validated dict, or a typed error

No module outside this package imports a concrete provider, and no module
outside `structured.py` parses model JSON (§16). Feature modules depend only
on `base.AIProvider` and `routing.TaskRouter`.
"""
