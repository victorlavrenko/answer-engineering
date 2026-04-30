"""Inference decode package.

Purpose:
    Group decode-loop and decode-session implementation concerns.

Architectural role:
    Internal package boundary for decode internals.

Import policy:
    Import directly from concrete decode modules.

Boundary note:
    Marker-only package surface reduces transitive imports at package load.

"""
