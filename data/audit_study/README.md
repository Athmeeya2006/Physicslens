# Physics audit at scale: data

How often does AI-generated video break physics, measured the same way on real footage of the same kinds of motion?

## Sources

| Set | Source | Licence | What is used |
|---|---|---|---|
| Generated | [VideoPhy-2 test set](https://huggingface.co/datasets/videophysics/videophy2_test) (clips from Hunyuan, CogVideo, VideoCrafter, Wan, Cosmos, Ray2 and Sora) | MIT | clips whose caption describes motion PhysicsLens can follow (drops, throws, bounces, swings, rolls, slides), spread evenly over generators and over the human physical-commonsense score (1 to 5) |
| Real (control) | [Physics-IQ](https://github.com/google-deepmind/physics-IQ-benchmark) full videos, public bucket `physics-iq-benchmark` | CC BY 4.0, Google DeepMind | 18 scenes with one object falling, rolling, swinging or colliding, filmed from three angles |

Nothing in this folder is a video: `manifest.json` lists every clip with its source URL, licence, caption and human score, so the selection can be checked and repeated (seed 7).

## Reproduce

```
python scripts/audit_study.py select --generated 400 --real 100   # rebuilds manifest.json (free)
python scripts/audit_study.py run --limit 500 --max-usd <cap>     # calls SAM 3.1 and Muse Spark Open
python scripts/audit_study.py report                              # tables and results.csv (free)
```

Every clip runs through the same pipeline a user's video does, with the same settings for generated and real clips. Each run saves what the audit read (`audit_input.npz`), so a changed rule is replayed on every stored run without another model call.

## Why the real clips matter

A physics lie detector is only as good as its false-accusation rate on real video. The real clips are the control: the rules are fixed so that real footage is not accused, and the same rules are then applied to the generated clips. The first real clips showed tracking artifacts at collisions (a Newton's cradle, a ball landing in sand) being read as violations; the contact rule in `backend/physicslens/physics/audit.py` came from that.
