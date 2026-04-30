"""Engine selection package.

Purpose:
    Group selection protocols and conflict-resolution implementation.

Architectural role:
    Internal package boundary for selection logic. This package does not define
    a convenience public facade.

Import policy:
    Import directly from concrete owning modules in this package.

Boundary note:
    Marker-only package surface avoids transitive imports during initialization.

"""
