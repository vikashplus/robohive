# Visual Encoders

RoboHive can render a camera and pass it through an encoder before it becomes part of the
observation, via the `visual_keys` mechanism (`MujocoEnv._setup(visual_keys=...)`, or set
through an env registration's `variants={'visual_keys': [...]}`, e.g. in
`robohive/envs/arms/__init__.py`). Implementation: `robohive/envs/env_base.py`,
`_setup_rgb_encoders()` (builds the encoder) and `get_visuals()` (applies it every step).

## Key format

```
rgb:<cam_name>:<H>x<W>:<encoder_id>
```

- `<cam_name>` — a camera defined in the robot config.
- `<H>x<W>` — the **capture resolution**: what's rendered (sim) or streamed (hardware) for
  that camera. This is independent of whatever size the encoder ultimately outputs.
- `<encoder_id>` — which encoder to apply to the captured image (see table below).

A companion depth key `d:<cam_name>:<H>x<W>:<encoder_id>` returns the raw depth map for the
same camera/resolution if also present in `visual_keys`.

All `rgb:` keys on a single env instance currently share **one** capture resolution and
**one** encoder — this is enforced by an assert in `_setup_rgb_encoders`. That's a scope
limitation of the current implementation, not a fundamental requirement (see Roadmap below).

## Supported encoders

| `encoder_id` | Output | Needs |
|---|---|---|
| `1d` | flattened raw pixels, `(H*W*3,)` uint8 | nothing |
| `2d` | raw pixels, `(H,W,3)` uint8 | nothing |
| `resize<H>x<W>` | bilinear-resized pixels, `(H,W,3)` uint8 | torch + torchvision |
| `crop<H>x<W>` | center-cropped pixels, `(H,W,3)` uint8 | torch + torchvision |
| `r3m18` / `r3m34` / `r3m50` | R3M embedding, `(D,)` | torch + torchvision + R3M |
| `rrl18` / `rrl34` / `rrl50` | ImageNet ResNet embedding, `(D,)` | torch + torchvision |
| `vc1s` / `vc1l` | VC-1 embedding, `(D,)` | torch + vc_models |

`resize`/`crop` follow torchvision's `T.Resize`/`T.CenterCrop` semantics exactly. Note
`crop<H>x<W>` **zero-pads** (rather than raising) if the requested crop is larger than the
captured `<H>x<W>` — that's `T.CenterCrop`'s native behavior, kept as-is for this minimal
implementation.

## Why a separate `resize`/`crop` encoder, instead of just setting capture `<H>x<W>`?

The capture `<H>x<W>` already lets you render at (almost) any size, so it's natural to ask
why a post-capture resize/crop step is needed at all. A few reasons it doesn't collapse into
"just pick the right capture size":

1. **Hardware cameras don't support arbitrary capture resolutions.** `Robot.get_visual_sensors`
   asserts a live camera's stream size exactly matches the requested capture size — real
   camera drivers/lenses typically support only a fixed set of native resolutions (often tied
   to calibration). If you want anything other than the native stream size in your
   observation, something has to resize it after capture.
2. **Resize and crop are different operations, not two spellings of the same thing.**
   Capturing at a smaller `<H>x<W>` resamples the *same field of view* at lower pixel density.
   It cannot select a sub-region of the scene (e.g. "zoom to where the hand operates"). A true
   crop requires capturing at the full resolution/FOV and cutting afterward — there's no
   capture-time parameter that does that.
3. **Decoupling capture config from per-task/per-model needs.** Capture resolution is often
   fixed by calibration or shared across many envs/tasks; different downstream policies want
   different input sizes from the same physical camera. `resize`/`crop` let one capture config
   serve many encoder targets without touching the capture-side config per task.
4. **Re-deriving multiple sizes from one recorded dataset.** If frames are recorded once at a
   native/high resolution, a resize/crop step (used offline) can derive different observation
   sizes after the fact without re-rendering or re-recording.

## Why torchvision, not OpenCV, for resize/crop

`torchvision` is already a lazily-guarded optional dependency for this exact feature (used by
the `r3m`/`rrl`/`vc1` transform pipelines already), so `resize`/`crop` add no new dependency.
It also operates on tensors, which keeps the door open for batched/GPU execution and for
chaining directly into a pretrained encoder's tensor pipeline with no extra numpy round-trips.
OpenCV would be a new, fairly heavy dependency whose default interpolation doesn't numerically
match torchvision/PIL's bilinear-with-antialias on downscale — for `rrl`/`r3m`, whose existing
`Resize(256)+CenterCrop(224)` preprocessing was tuned against torchvision, swapping in cv2
resize could silently shift the input distribution their checkpoints were trained on.