"""Engine scoring package.

Purpose:
    Group scoring interfaces and implementations.

Architectural role:
    Internal package boundary for scoring behavior. This package does not define
    a convenience public facade.

Import policy:
    Import directly from concrete owning scoring modules.

Boundary note:
    Keeping this package marker-only reduces duplicate import surfaces.

"""
