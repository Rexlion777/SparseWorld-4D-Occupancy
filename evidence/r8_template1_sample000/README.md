# R8 Template 1 sample 000 provenance

This directory contains rendered semantic-Occupancy anchors for one frozen
`A10_drop_front_triplet` sample:

- `native_h{0,2,4,6}.png`: degraded-native `native_semantic` outputs;
- `r8_h{0,2,4,6}.png`: raw SW13A/R8 `teacher_raw_semantic` outputs;
- `a10_front_triplet_camera_panel.png`: the matching six-camera observation
  panel with the failed front triplet blacked out.

The semantic anchors were exported read-only from the frozen teacher cache for
sample 000. `teacher_raw_semantic` is the direct R8 output before later
post-processing. Ground truth, the current clean image, and future frames are
not used by the repair path.

The GIF/MP4 generator performs smooth display interpolation between the genuine
0/2/4/6 s anchors. Interpolated frames are visualization frames, not additional
model predictions.

To rebuild the public videos from these anchors:

```bash
python scripts/portfolio/generate_r8_template1_comparison.py \
  --anchors evidence/r8_template1_sample000 \
  --output-dir assets/template1/r8_comparison
```
