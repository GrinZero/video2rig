# How it works

## The idea

Frame 0 of a generated clip is our own render: the pose and the cameras are known exactly. So before any clip, the
tool can learn what this one creature looks like through these cameras, with exact labels and nothing drawn by
hand. It renders the creature in thousands of random poses and trains a small network, the joint reader, to find
every joint in such a picture. Then it reads every joint of a clip, in every frame and every view, and fits the rig
to those joints.

1. **Train a reader** (once per creature: `python -m video2rig.reader.prepare`, docs/WORKFLOW.md step 2b).
   - 3000 random poses of the rig within its joint ranges: the body turned, shifted or placed anywhere the start
     frame leaves room, on the floor or in the air, the jaw open in half of them, two limb ends brought together or
     one laid on the body. In about half of them the bones move during the frame, so the render has motion blur,
     sometimes with the whole body lunging or spinning. No animation goes in: the reader learns the creature, not a
     move, so one reader reads every move.
   - Blender renders every pose through the start frame's own cameras. The tool works out which joints each picture
     shows (the posed mesh in a depth buffer): a far leg behind the near one or an elbow behind the chest is
     hidden, and the reader is taught to say "not here" there instead of guessing.
   - The network: DINOv2-small (Meta, Apache 2.0) sees the picture at 448 px, a small decoder draws one heat map per
     joint. Every training picture is damaged the ways a generated frame differs from our render: re-framed, other
     colours and light, blur, a fast limb smeared into a see-through streak (Seedance draws a strike as a 30-100 px
     smear), lower resolution, noise, a shadow and a tinted floor. It is never mirrored: left and right are exactly
     what it must tell apart.
   - 5% of the poses are held out. Their error in pixels says whether the reader can be trusted (2-4 px for the
     readers in use). Longer training helps up to about 9000 steps.
2. **Prepare the clip.** Frames, the creature's silhouette in every frame, the content frame rate (Seedance clips
   drop frames, so one step in four can carry twice the motion), and the cameras aligned on frame 0 (Seedance
   re-frames the start image a little: scale, offset, sometimes a turn of 10+ degrees).
3. **Read.** The reader finds every joint in every frame and view, and says how sure it is (the peak of its heat
   map: about 1 where it sees the joint plainly, low where it guesses). It reads every picture 8 times, each copy
   shifted by a few pixels and scaled a little, brings the copies' heat maps back onto the picture and averages them:
   a reader that reads each frame on its own errs from frame to frame most on small, fast parts (hands, elbows,
   paws), and an error that does not repeat between the copies shrinks in their average (the arms' frame-to-frame
   error -34 to -47% on 30 clips). Half a minute on the graphics card for a 5 s clip. A prop is
   read as a few points of its own (a log's ends, a round stone's middle) and fitted as a rigid body.
4. **Left / right check.** A side view can take the near leg for the far one; the front view sees them apart. For
   each pair of twin limbs the side view's readings are kept or swapped, whichever lets both views agree on one 3D
   point per joint. It is decided along the whole clip, not frame by frame: every change of mind costs, so one
   unclear frame does not swap a leg. A reading that jumps to the twin limb for a frame is put back, a short unsure
   stretch (a hand behind the body) is bridged between the sure readings around it, and a far limb's guess that the
   other view contradicts counts little.
5. **Fit, frame by frame.** In every frame, the pose of the rig whose joints land where the reader saw them in every
   view: each reading weighted by how sure the reader was, far misses barely pulling (they are wrong readings),
   within the rig's joint ranges (knees and elbows bend one way), nothing under the floor, a planted foot flat. Each
   frame starts from the frame before (frame 0 from the start pose, which the clip's first frame shows) and from a
   few other starts (the limbs in the start pose, the limbs where both views read them); the best fit wins. A jaw
   the reader sees opens for a bite. The fit runs on one processor thread, so a clip gives the same motion on every
   computer.
6. **Clean-up**, as in motion capture: one-frame spikes smoothed unless the reading backs them (a galloping paw
   really turns back 15-20 degrees in one frame), the body's turn smoothed as a turn, and a low-pass at 8 Hz. A bone
   that moves slowly all through the clip gets 4 Hz, or 5 Hz if it still has one fast moment (a landing, a strike):
   the reader reads each frame on its own and its frame-to-frame error shows most on small parts (hands, elbows,
   paws) as a tremble. A limb end the low-pass put under the floor bends out by the least change to the smoothed
   pose.
7. **Feet, steps and loose parts.** Seedance often draws a walk or a run on the spot: the body stays and the standing
   feet slide back, in the video too. When the low feet slide back the same way for most of the clip, the body is
   moved forward by what they slide, so the creature walks and its feet stand (`steps`). Then every foot that is down
   and still is pinned where it stands by the least change of its own leg (`foot_lock`; the heel may lift). Last,
   tails and ears follow through a damped spring instead of shaking with the reading (`springs`; not wings, whose
   flap is the motion itself). Each is one switch (`--set steps=false` ...).
   A walk moved by its steps travels where the video stays, so in the review film its outline drifts off the video's.
8. **Arms out of the body.** A hand one camera cannot see is guessed, and the guess went into the creature's own
   trunk (a centaur's sword hand in its horse's flank). Points on every free limb (its keypoints, the middle of each
   bone, and the blade of what a hand holds) are kept out of capsules round the trunk's bones, in the fit and once
   more after the clean-up (`no_self_pass`).

Then the motion is written for Blender: every bone's rotation at the clip's real times, the root's position and every
prop's matrix.

On the bench (clips rendered from known motions of nine creatures, through their own cameras) the fitted bones point
within about 3 degrees of the truth (median per bone). The older road below gave 21 degrees on the wolf's gallop.

Before the fit, `python -m video2rig check` uses the same reading to measure how far the two views disagree. When
the generator drew a different motion in each view (the body more than 1.5% of its height apart), no fit can make one
true motion out of them: generate the clip again.

## What it can and cannot see

- Two views see depth; one view guesses it from the start pose and the skeleton. Use two views for arms crossing
  the body, rolls, turns towards the camera.
- A limb behind the body in one view is read from the other view. Hidden in both, it is guessed, and those guesses
  are the reader's weakest point (a limb behind the body in the side view: 5-12 px off in our tests).
- The tool reproduces the video, including its mistakes: a sliding foot, a floating jump, a prop changing size, two
  views that do not agree.
- Fingers are not fitted (480p cannot resolve them); they keep the start pose. Curl them in Blender if needed.
- A held weapon's direction is not read: the hand's twist leaves no mark on the wrist and hand-tip points, so the
  blade can point the other way than in the video.
- A reader knows one creature through one set of cameras. A new model, rig or start frame needs a new reader.

## The older road (`--old`)

Before the reader, the tool followed points instead of joints. `run` still takes this road when the creature has no
`reader/reader.pt`, or with `--old`. Seeds on the model's visible pixels in the start frame (dense on limbs and
props) are followed through the clip by BootsTAPIR (Google DeepMind, Apache 2.0). Then one optimisation over all
frames finds the joint angles that put every tracked point on its track, while the model stays inside the video's
silhouette, joints stay in range and the motion stays smooth; when a body rolls over and the outline is lost, it
tries the body turned over. It needs no training, but a clip takes 1-3 minutes of tracking and 5-15 of solving, it
has many hand settings (`solve.*`, docs/CONFIG.md), and on the bench's wolf gallop it was 21 degrees per bone off
(the reader: about 3).

## Why not the usual ways

- Fitting the outline frame by frame (what we did first): the outline of a near leg and a far leg is the same, so
  legs swapped and arms twisted far past what a joint can do; every clip needed hours of hand fixes.
- Pose detectors (2D keypoints): made for people, swap left and right in side views, nothing for animals. The joint
  reader is a pose detector too, but trained on this one creature through these cameras, with exact labels.
- Video mocap services: humans only, and they lose arms hidden by a prop.
