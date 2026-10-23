# Making the video: clips the tool can read

An image-to-video generator animates one still frame into a 4-8 s video. Any generator that starts from a given
first frame and keeps it can be used (Seedance, Kling, Veo, Runway, Hailuo...). Here the video is a motion reference:
what matters is that the first frame is exactly our render and that the body stays readable.

Everything below was made and measured with Seedance (ByteDance: Dreamina "Seedance 2.x", Topview, the BytePlus API),
the generator we tested. The rules for the start picture, the prompt and checking the clip apply to any generator;
the API road and the pack ledger are Seedance's own.

## Where to generate

- **BytePlus ModelArk** (the official Seedance API): `python -m video2rig.generate <project> <clip> --prompt <file>`
  sends the project's start frame as the first frame and saves `clips/<clip>.mp4`. Paid per token (720p 5 s with
  sound: 108,900 tokens), no app queue, several variants at once (`--variants 3`). Key from `BYTEPLUS_API_KEY` or
  `~/.byteplus-key`; activate the Seedance model in the console first. A 720p 5 s clip was ready in about 5 minutes.
  Options: `--start` (another first frame), `--resolution 480p|720p|1080p` (default 720p), `--duration` (seconds,
  default 5), `--model`.
- **Dreamina** (the app): "First and last frames" mode with only the first frame. Its queue can hold a clip for an
  hour at busy times.

### The pack ledger (BytePlus only, required)

`generate` refuses to start without `~/.byteplus-pack.json` (in PowerShell: `$HOME\.byteplus-pack.json`). The file
says how much money the tool may spend, so nothing is billed by surprise:

```json
{"remaining_pack_ktokens": 674.0, "pack_per_api_token": 1.67, "allow_pay_per_use": false}
```

| Field | What it means |
| --- | --- |
| `remaining_pack_ktokens` | thousands of tokens left in your BytePlus resource pack (from the ModelArk console). Required |
| `pack_per_api_token` | pack tokens charged per API token (1.67 measured on 2026-10-09; the default) |
| `allow_pay_per_use` | `true`: a clip that does not fit in the pack still goes through and is billed per use. Default `false` |

Before sending, the tool estimates the cost (about 9,700 / 21,700 / 48,800 API tokens per second at 480p / 720p /
1080p, times the variants) and refuses when it needs more than 97% of what is left and pay-per-use is off. After
each clip it takes the clip's real tokens off and writes the time in `updated`.

**No resource pack (pay per use only):** write

```json
{"remaining_pack_ktokens": 0, "allow_pay_per_use": true}
```

From PowerShell, without a byte order mark (the tool cannot read a file that starts with one):
`Set-Content -Encoding ascii "$HOME\.byteplus-pack.json" '{"remaining_pack_ktokens": 0, "allow_pay_per_use": true}'`.
The remaining count then goes below zero as you spend; that is fine.

### A run that stops

The task ids go to `clips/<clip>.task.json` the moment BytePlus accepts them: a paid task goes on at BytePlus even
if your computer goes to sleep or the network drops. While waiting the tool tries again after network errors (15 s,
then doubling up to 5 minutes, 8 times in a row). If it still stops, it prints the command that brings the clips
home without paying again:

```powershell
python -m video2rig.generate my_wolf run01 --resume
```

It downloads the clips, writes `clips/<clip>.json` and updates the ledger; the task file is deleted when every clip
in it is home. A new run of the same clip name refuses while the task file is still there.

## The start frame

1. **Always our own render** of the rigged model (`blender/render_start.py`), never a drawing or an AI image: frame 0
   of the video = a known pose under a known camera. That is what everything else stands on.
2. **Two views in one frame** (front | side) for anything with arms or legs crossing the body. Seedance keeps both
   panels in sync, and the second camera shows the depth the first cannot.
3. Plain grey studio, visible floor and soft shadow, the whole body in frame with room around it for the move.
4. **Moves that travel forward fast (a leap, a lunge): the stacked frame** (`blender/render_stack.py`): the side view
   above, the view from above below, one scale, lined up, so the forward axis runs the same way in both strips (one
   image column = one place along the body, as one image row is one height in front | side). Side | top panels side
   by side drew forward motion horizontally in one panel and vertically in the other, and Seedance let them drift
   apart: a boss's leap disagreed by up to 1.6 m in the air (check 6.1%); the same prompt stacked agreed to about
   0.1 m in the air (check 1.8%, limbs 4.8% -> 1.7%), its landing spot still 0.4 m apart. The strip from above is
   turned upright when it is read (`turn` in cameras.json) and both strips are read through windows that follow the
   body (`read_window`, from `--read-like` the side | top frame the reader learned), so the same reader reads it.

## Resolution, length

- One view: 480p is enough. Split frame (two views): 720p. 1080p only for small details.
- 4-5 s per move; 8 s for a combo. One move per clip.
- For loops, ask for the end pose to be the start pose.

## No effects, ever

End every prompt with this sentence (`python -m video2rig.generate` adds it by itself; in Dreamina or Topview paste it):

> No visual effects of any kind: no dust, no smoke, no debris, no sparks, no particles, no speed lines, no motion
> trails, no light trails, no motion blur, no camera shake. The blade and every limb stay sharp, the flat floor stays
> perfectly clean, and every foot, hand and held object stays clearly visible in every frame.

Seedance likes to add a dust cloud at every hoof strike and sparks and a light trail to every sword cut. The joint
reader learned on clean renders: under the dust it does not see the hooves, yet it reads them with full confidence,
and the hooves come out wrong. A centaur's combo regenerated with this sentence: no dust, no sparks, the hooves visible
all the time (only faint trails behind the fastest cuts stayed).

## Prompt wording that worked

- Describe the creature as it is in the image, then the move, then what EACH camera sees at the END ("in both views
  it ends lying on its back, legs up"). Asked only "on its back", Seedance rolled it back onto its belly.
- "Static cameras, no zoom, no cuts, no camera shake, no slow motion, no motion blur." Keep "the plain grey studio,
  the level floor and the soft constant lighting".
- Props: "The log is one rigid, straight log of constant length and thickness: it never bends, shortens, splits,
  duplicates, and it never passes through his body, head or hands." Say where it lies until it is picked up ("the
  log lies still on the ground until his hands lift it").
- A hit or blow: "convulses as if hit, with no visible cause: no object, weapon, hand, shockwave, ripple, flash, dust
  or any other visual effect ever appears, and the image stays sharp". "Struck" or "hit" alone draws a club; "an
  invisible blow" draws a shockwave.
- Fast backward moves overshoot: leave room behind the creature and say "the body keeps its shape".
- Heavy creatures float (a leap hangs in the air, a rearing creature lifts its back feet): fix it in the Action
  (retime the hang, lower the body), not by generating again.
- Keep "keep the whole body inside the frame the whole time" in every prompt.

## Check the clip before using it

- Both panels move together (same phase in every frame).
- Nothing crosses the split line; the body never leaves the frame.
- No invented objects or effects; props keep their size.
- The end pose is what you asked for.

Reject and regenerate rather than fit a bad clip: the tool reproduces the video, mistakes included.

## Timing

Seedance clips are not what their file says: 24 fps content, or 30 fps with every 5th frame dropped, or 25 with
every 25th dropped, in a 24, 30 or 60 fps container. The tool measures it per clip (`work/<clip>/timing.json`) and keys
the Action at the real times, so the motion does not speed up and slow down every few frames.

## Sound

The clips come with sound (steps, grunts, impacts). Keep the files: the sound fits the motion you get, cut it at the
same places as the Action.
