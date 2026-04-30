"""Engine orchestration stages package.

Purpose:
    Group stage handlers used by runtime orchestration.

Architectural role:
    Internal package boundary for stage implementations. This package does not
    define a convenience public facade.

Import policy:
    Import stage classes directly from concrete stage modules.

Boundary note:
    Marker-only package surface reduces import coupling between stages.

"""
