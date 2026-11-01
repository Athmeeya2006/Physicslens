"""PhysicsLens's own solar-system simulator: Newtonian N-body with optional corrections, validated against NASA/JPL.

bodies (the catalogue), ephem (cached JPL states and masses), nbody (the integrator and forces), elements (state
vectors to orbital elements), validate (comparisons with JPL). Everything runs offline from data/sky/; fetch it once
with scripts/sky_fetch.py.
"""
