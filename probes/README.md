# Probes — the witness scripts (`development_principles.md` §2)

**The contract:** a probe is a script in this directory, named after the mechanism
it proves (`probe_<mechanism>.py`), that drives that mechanism **end to end
against the real running deployment** — the Docker stack, real HTTP, real
Postgres, a real model call where one is involved — and prints a verdict a
human can read.

A probe is not a unit test:

- It talks to `http://localhost:8000` like the app does, never to functions in-process.
- Its output IS the observation §1 demands. Reading a green verdict — actually
  reading it, not skimming the exit code — is what lets you say "witnessed".
- It must be runnable on demand, forever. When a mechanism changes, its probe
  changes in the same commit.

The minimum probe set (one per locked mechanism) and the step that delivers
each is listed in `IMPLEMENTATION_PLAN.md` and tracked in `PICKUP.md`.
None exist yet beyond this README.
