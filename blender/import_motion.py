"""Put a video2rig motion onto the rig in Blender: an Action on the armature (and the props' motion).

In Blender's Scripting tab with your .blend open: set MOTION below and run. Or without a window:
    blender model.blend --background --python blender/import_motion.py -- --motion my_creature/out/clip.npz --save out.blend

Keys land at the clip's real (content) times at the scene's frame rate, starting at the scene's start frame; every
bone that the fit did not move keeps the start pose. Rotations are quaternions, linear between keys, each key on the
same hemisphere as the one before (no flips when an engine samples between keys).
"""
import argparse
import os
import sys

import bpy
import numpy as np
from mathutils import Matrix, Quaternion, Vector

MOTION = ''          # path to the .npz when run from the Text Editor


def args():
    argv = sys.argv[sys.argv.index('--') + 1:] if '--' in sys.argv else []
    ap = argparse.ArgumentParser(prog='import_motion.py')
    ap.add_argument('--motion', default=MOTION)
    ap.add_argument('--armature', default='')
    ap.add_argument('--action', default='')
    ap.add_argument('--save', default='')
    return ap.parse_args(argv)


def action_fcurves(act):
    """Every F-curve of an action. Blender 4.x: action.fcurves. Blender 5.x has slotted actions only (Action.fcurves
    is gone): the curves sit in action.layers[i].strips[j].channelbags[k].fcurves."""
    fcurves = getattr(act, 'fcurves', None)
    if fcurves is not None:
        return list(fcurves)
    out = []
    for layer in getattr(act, 'layers', ()):
        for strip in layer.strips:
            for bag in getattr(strip, 'channelbags', ()):
                out.extend(bag.fcurves)
    return out


def main():
    a = args()
    if not a.motion:
        sys.exit('import_motion: give --motion <file.npz> (or set MOTION in the script)')
    edit = bpy.context.preferences.edit
    was = edit.keyframe_new_interpolation_type
    edit.keyframe_new_interpolation_type = 'LINEAR'     # new keys linear; set back below, whatever happens
    try:
        insert(a)
    finally:
        edit.keyframe_new_interpolation_type = was


def insert(a):
    d = np.load(a.motion)
    scene = bpy.context.scene
    arms = [o for o in scene.objects if o.type == 'ARMATURE']
    arm = scene.objects[a.armature] if a.armature else arms[0]
    fps = scene.render.fps / scene.render.fps_base
    frames = scene.frame_start + d['times'] * fps
    name = a.action or os.path.splitext(os.path.basename(a.motion))[0]
    act = bpy.data.actions.new(name)
    act.use_fake_user = True
    arm.animation_data_create()
    arm.animation_data.action = act
    bones = [str(b) for b in d['bones']]
    root = str(d['root'])
    last = {}
    for pb in arm.pose.bones:
        pb.rotation_mode = 'QUATERNION'
    for i, f in enumerate(frames):
        for j, bname in enumerate(bones):
            pb = arm.pose.bones.get(bname)
            if pb is None:
                continue
            q = Quaternion(d['quaternions'][i, j])
            if bname in last and q.dot(last[bname]) < 0:
                q.negate()
            last[bname] = q.copy()
            pb.rotation_quaternion = q
            pb.keyframe_insert('rotation_quaternion', frame=float(f))
            if bname == root:
                pb.location = Vector(d['root_location'][i])
                pb.keyframe_insert('location', frame=float(f))
    # props: X_t = M_t X_start in armature space -> world matrix W_t = A M_t A^-1 W_0
    A = Matrix(d['armature_world'].tolist())
    for k, pname in enumerate(str(p) for p in d['props']):
        obj = bpy.data.objects.get(pname)
        if obj is None:
            print(f'import_motion: prop {pname} not in the scene, skipped')
            continue
        W0 = obj.matrix_world.copy()
        obj.animation_data_create()
        obj.animation_data.action = bpy.data.actions.new(f'{name}_{pname}')
        obj.rotation_mode = 'QUATERNION'
        for i, f in enumerate(frames):
            obj.matrix_world = A @ Matrix(d['prop_matrices'][i, k].tolist()) @ A.inverted() @ W0
            obj.keyframe_insert('location', frame=float(f))
            obj.keyframe_insert('rotation_quaternion', frame=float(f))
    for o in [arm] + [bpy.data.objects.get(str(p)) for p in d['props']]:
        if o is None or o.animation_data is None or o.animation_data.action is None:
            continue
        for fc in action_fcurves(o.animation_data.action):
            for kp in fc.keyframe_points:
                kp.interpolation = 'LINEAR'
    scene.frame_end = max(scene.frame_end, int(np.ceil(frames[-1])))
    print(f'IMPORT_MOTION {name}: {len(frames)} keys, {frames[-1] - frames[0]:.1f} frames at {fps:g} fps', flush=True)
    if a.save:
        bpy.ops.wm.save_as_mainfile(filepath=os.path.abspath(a.save))
        print('IMPORT_MOTION saved', a.save, flush=True)


main()
