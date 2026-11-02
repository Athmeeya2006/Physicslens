# New occultations: sources checked (2026-11-02)

No reduction and no pre-registration: no source found publishes recent (2025 to 2026) observer timings openly, with
terms that allow reuse, before an official reduction exists.

| Source | What it offers | Why not used |
|---|---|---|
| Euraster (euraster.net), European asteroidal occultation results | per-event chords and profiles | the domain now redirects to an unrelated site (HTTP 301 to another host); not followed. When it worked, it posted its own fitted profile with the chords, so there was no unreduced window to predict into |
| IOTA North American results (asteroidoccultation.com/observations) | results pages | the reports index found is for 2019 ("profiles provided here are preliminary"); the 2025 and 2026 data folders return 404; no reuse terms stated |
| SODIS (sodis.iota-es.de), the IOTA/ES report system | observer reports | listing behind login and register links; footer "All Rights Reserved"; no data licence |
| Occult Watcher Cloud | predictions and observer reports | account based; no open data terms found |
| Lucky Star project | TNO occultations | results appear in papers, already reduced |
| NASA PDS "Asteroid Occultations V5.0" | archived chords | already reduced and not recent |

The chord-to-ellipse reduction is ready (`backend/physicslens/astro/occultation.py`, used by
`scripts/occultation_demo.py`). Its recovery of a known ellipse from synthetic chords is covered by
`tests/test_occultation.py` (`test_ellipse_axes`, `test_a_miss_bounds_the_outline`).
