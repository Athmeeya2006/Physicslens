# Filming checklist for four new gallery results

Four real clips that would each show a capability no current gallery result shows: drift between two recordings of
one machine, restitution per bounce, a spin-down friction law, and a 3D throw. Each analysis is a normal upload (SAM
3.1 and Muse Spark Open, through the cost guard); everything after it (comparison, gallery, exports) is free.

## Before any of them

- [ ] Phone on a support (a tripod, or a stack of books with the phone taped to it), not in the hand.
- [ ] On the Analyze page, watch the level: it should say "Level" (within 3°) before you press record, then press
      "Save this tilt with the clip".
- [ ] Camera app: 1080p, **60 fps** (30 fps if 60 is not offered), no HDR, no "cinematic" or portrait mode, no
      stabilisation if the app lets you turn it off (it shifts the frame).
- [ ] Lock focus and exposure (tap and hold on the object until the lock icon shows).
- [ ] Bright, even light, no flicker: daylight near a window, or several room lights. Avoid a single bare bulb behind
      the object.
- [ ] A plain background that contrasts with the object (a wall, a sheet).
- [ ] An A4 sheet taped flat **in the plane of the motion** when lengths matter (rows below say so).
- [ ] Start recording, wait 1 s, then start the motion; keep recording 1 s after it ends. 5 to 20 s in total.

## 1. Door closer, twice, a week apart (machine-health drift)

- [ ] Phone 1.5 to 2 m away, level, looking straight at the door from the hinge side's opposite wall, so the door's
      free edge moves across the frame. Put a bright sticker (or a strip of coloured tape) on the door's edge near the
      handle: that is what will be tracked.
- [ ] Open the door to the same angle each time (mark the floor with tape), let go, film until it closes.
- [ ] Record three releases now, and three more at least seven days later, same marks, same phone position.
- [ ] Upload each with the hint "the sticker on the door edge".
- [ ] On the Health page, compare a run from week 1 with one from week 2 (or all six in the trend view): the result
      says whether damping or frequency changed beyond the measurement's own spread.
- [ ] Add the clearest pair to the gallery (commands below).

## 2. Bouncing ball (restitution per bounce)

- [ ] A tennis ball or a rubber ball; dropped (not thrown) from about 1 m onto a hard floor.
- [ ] Phone 1.5 m away, side-on, the camera at the height of the middle of the fall, the whole fall and at least three
      bounces in frame.
- [ ] A4 sheet taped flat on the wall directly behind the bounce line, or enter the ball's diameter as the object size.
- [ ] **120 or 240 fps slow motion** if your phone has it (the impacts are short); enter the slow-motion factor on
      upload (8 for 240 fps played at 30).
- [ ] Ground truth to write down first: the drop height (tape measure).

## 3. Spinning fidget toy slowing down (friction law)

- [ ] Phone directly above the toy, looking straight down, 25 to 40 cm away, level (the level should read "Level" with
      the phone flat). The toy on a table on its base.
- [ ] A small mark (a dot of correction fluid or tape) on one arm, so its angle can be read.
- [ ] **240 fps slow motion** if available; a spinner turns faster than 30 fps can follow.
- [ ] Flick it hard, film until it has nearly stopped (often 30 s or more: trim to the last 20 s, or to the part
      where the mark is still sharp).
- [ ] Optional: enter the number of arms as the blade count, so the sound of the spin is converted to the rotor's
      own rates too.

## 4. A 3D throw toward the camera (depth mode)

- [ ] Calibrate the lens first on the Analyze page (print `/checkerboard.svg` at 100%, film it for 15 s, Calibrate).
      Use the same camera setting for the throw.
- [ ] A ball of known diameter (a tennis ball is 6.7 cm); measure it.
- [ ] Phone level on a support 2.5 to 3 m from the thrower, at chest height; save the tilt.
- [ ] Throw gently in an arc that comes toward the camera (from about 3 m away to about 1.5 m), never filling more than
      a third of the frame and never leaving it. Plain background.
- [ ] Upload with the ball's diameter as the object size and "Also measure a thrown ball's depth (3D)" ticked.

## Adding a result to the gallery

After an analysis has finished, its run id is in the page address (`?run=<id>`). With the run stored under
`var/runs/<id>`:

```bash
python scripts/add_to_gallery.py <run id> --id gallery-doorcloser --order 6 \
    --title "Door closer: a week of wear in its damping" \
    --description "Two recordings of the same door closer a week apart ..." \
    --credit "Video: filmed for PhysicsLens, CC BY 4.0"
python scripts/add_to_gallery.py <run id> --id gallery-bounce --order 7 --title "..." --description "..." --credit "..."
python scripts/add_to_gallery.py <run id> --id gallery-spinner --order 8 --title "..." --description "..." --credit "..."
python scripts/add_to_gallery.py <run id> --id gallery-throw3d --order 9 --title "..." --description "..." --credit "..."
python scripts/build_static_gallery.py          # the static site's copy, share pages and the dataset
python scripts/baselines.py                     # classical baselines on every gallery track, including the new ones
```

Add each clip's licence to `gallery/README.md`, and its truth (if one was measured) to `TRUTHS` in
`scripts/baselines.py`.
