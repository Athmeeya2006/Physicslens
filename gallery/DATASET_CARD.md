# PhysicsLens open dataset: physical parameters measured from video

One row per physical parameter that PhysicsLens measured from a video and then verified: the law it belongs to forecast
motion it was never fitted on. The table is built from the committed gallery results by
`scripts/build_static_gallery.py` and served at `/gallery/dataset.json` and `/gallery/dataset.csv` on the site, with
Croissant metadata at `/gallery/croissant.json`.

## Fields

| Field | Type | Meaning |
|---|---|---|
| `id` | text | `<run id>-<property>`, unique |
| `object` | text | what was tracked, as the scene model named it; "central body" for a gravitational parameter, which belongs to the body being orbited |
| `property` | text | a fixed vocabulary: `gravity`, `natural_frequency`, `damping_ratio`, `joint_damping`, `drag_rate`, `drag_coefficient`, `friction_dry`, `friction_viscous`, `gravitational_parameter` |
| `meaning` | text | the property in words, for this law |
| `value` | number | the fitted value |
| `std` | number | one standard deviation, from residual block bootstrap refits that include the scale's own error |
| `units` | text | SI unless the row says otherwise (orbits: AU and days, or Jupiter radii and hours) |
| `family` | text | the motion family (pendulum, oscillator, projectile, rotation, plane) |
| `model` | text | the law's id; `custom_*` is a law the agent wrote, parsed by the closed grammar |
| `heldout_error` | number | RMSE of the law's forecast over the last quarter of the motion, fitted on the first three quarters, divided by the motion's standard deviation |
| `scale_basis` | text | where lengths came from: `user` (a length given by the person filming), `object_size` (the object's size given), `known_scale` (known from how the video was drawn), `gravity` (g = 9.81 m/s² as the ruler), `none` (the parameter needs no scale), or the kind of reference object in view (for example `a4_sheet`) |
| `conditions` | text | what was filmed and how |
| `run_id` | text | the gallery result the row comes from (open it at `/?run=<run id>`) |
| `clip` | text | the clip's title |
| `license` | text | the clip's source and licence |

## How each value was verified

A row exists only when all of these held for its result (`backend/physicslens/export/dataset.py`):

1. The result was **accepted**: the law, fitted on the first 75% of the motion, forecast the last 25% with a held-out
   error under 0.15, beat every simpler law by BIC with the effective sample size, and also forecast from four later
   starting points (rolling origins at 55, 65, 75 and 85% of the clip).
2. A referee model proposed up to three alternative laws and none of them did better by the same rule.
3. The parameter was **measured**: not given by the user (a pendulum's length is never a row), not in arbitrary units
   (a clip with no scale gives frequencies and damping only), not derived from other rows.
4. It has an uncertainty from the bootstrap.

Values describe the object under the conditions it was filmed in (a damping rate depends on the air and the joint),
so every row carries its clip, conditions and run.

## Known limits

- Six results, so far fewer rows than a reference table; each is a worked example with its full decision trail, not a
  survey.
- Three rows come from videos rendered from NASA/JPL Horizons positions (Apophis, Halley, Io), not from a camera. Their
  truths are known exactly, which is why they are there: they test the measurement chain, not the camera.
- 2D motion in the camera plane only.

## Licence

The table (the measured values and their metadata) is released under
[CC BY 4.0](https://creativecommons.org/licenses/by/4.0/): use it with credit to PhysicsLens. The source clips keep
their own licences, named in each row's `license` field and in `gallery/README.md` (two are CC BY-SA 4.0 from
Wikimedia Commons; three are renderings of public NASA/JPL data). The code that produced it is MIT licensed.

## Citing it and minting a DOI (Zenodo)

1. Build the files: `python scripts/build_static_gallery.py` writes `frontend/public/gallery/dataset.json`,
   `dataset.csv` and `croissant.json`.
2. On [zenodo.org](https://zenodo.org), sign in, choose **New upload**, and add `dataset.csv`, `dataset.json`,
   `croissant.json` and this card.
3. Resource type **Dataset**; title "PhysicsLens open dataset: physical parameters measured from video"; licence
   **Creative Commons Attribution 4.0**; add the repository URL under **Related works** ("is supplemented by").
4. Publish. Zenodo assigns the DOI; add it to this card and to the README's dataset section.

To version it with the code instead, connect the GitHub repository in Zenodo's GitHub settings: every GitHub release
is then archived with its own DOI.
