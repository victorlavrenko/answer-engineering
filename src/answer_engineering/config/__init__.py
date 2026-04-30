"""Configuration defaults grouped by owning subsystem.

Purpose:
    Mark configuration modules as a deliberate package-level boundary rather
    than a miscellaneous constants bucket.

Architectural role:
    Namespace for default policy objects used by runtime orchestration, scoring,
    and rule matching.

Inputs (architectural provenance):
    Imports are expected to originate from concrete configuration modules that
    own a specific subsystem's default behavior.

Outputs (downstream usage):
    Supports stable package discovery for callers and tests that need to locate
    configuration surfaces.

Invariants/constraints:
    Keep configuration ownership explicit. Avoid turning this package into a
    compatibility layer or a broad re-export surface.

"""
