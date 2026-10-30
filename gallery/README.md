# Gallery

Curated results of real PhysicsLens analyses, served read-only by the app.

Each entry holds `result.json` (the analysis), `clip.mp4` (the analyzed clip), `share.mp4` (the clip with
the trail, ghost and equation burned in) and `meta.json` (title, description, credit).

## Licenses of the clips

The code in this repository is MIT licensed. The video clips are not covered by that license:

- `gallery-apophis`: rendered by PhysicsLens (`scripts/orbit_demo.py`) from NASA/JPL Horizons state vectors of
  99942 Apophis and Earth; no third-party footage. Data credit: NASA/JPL Solar System Dynamics (Horizons, CNEOS).
- `gallery-halley`: rendered by PhysicsLens (`scripts/halley_demo.py`) from NASA/JPL Horizons state vectors of
  comet 1P/Halley (record 90000030) and Earth, 1985 to 1987; no third-party footage. Data credit: NASA/JPL Solar System Dynamics (Horizons).
- `gallery-jupiter`: rendered by PhysicsLens (`scripts/jupiter_demo.py`) from NASA/JPL Horizons state vectors of
  Io around Jupiter; no third-party footage. Data credit: NASA/JPL Solar System Dynamics (Horizons).
- `gallery-pendulum`: "Schwingung (Definition)" by Rhetos / Mathe-AC Lernwerkstatt, from
  [Wikimedia Commons](https://commons.wikimedia.org/wiki/File:Schwingung_(Definition).webm), licensed
  [CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/). Changes: trimmed to 16.0 to 22.5 s and
  downscaled; `share.mp4` adds an overlay. These derived clips are shared under the same license.
- `gallery-paperbag`: "Ballistische Kurve" by Rhetos / Mathe-AC Lernwerkstatt Aachen, from
  [Wikimedia Commons](https://commons.wikimedia.org/wiki/File:Ballistische_Kurve.webm), licensed
  [CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/). Changes: the 20x slow-motion part (filmed at
  200 fps) trimmed to 15.6 to 33.4 s and re-encoded at 30 fps; `share.mp4` adds an overlay. Shared under the same
  license. Analyzed with a slow-motion factor of 20, so times and rates are real time.
- `gallery-spring`: "Oscillatory motion acceleration" by Pieter Kuiper, Linnaeus University, from
  [Wikimedia Commons](https://commons.wikimedia.org/wiki/File:Oscillatory_motion_acceleration.ogv), under the
  Wikimedia "Attribution" license (free use with credit). Changes: trimmed to 3.0 to 18.0 s and re-encoded;
  `share.mp4` adds an overlay.
