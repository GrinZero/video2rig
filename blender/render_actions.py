"""Film a creature's animations, one after another, in render_start.py's grey studio from a three-quarter camera
(the floor and the shadows show, so a foot that slides or sinks shows too): to look at the motion itself, without
the video beside it.

    blender model.blend --background --python blender/render_actions.py -- --out anims.mp4
        [--actions run01_reader,walk01_reader] [--labels GALLOP,WALK] [--width 1280] [--height 720] [--samples 16]

Every action of the armature by default (those blender/import_motion.py made). Frames go to a folder beside the
film, the film is joined by ffmpeg (on PATH) with each action's label burned in.
"""
import argparse
import math
import os
import shutil
import subprocess
import sys
import traceback

import bpy
from mathutils import Vector

GREY = (0.42, 0.41, 0.40, 1.0)
FONT = r'C\:/Windows/Fonts/arial.ttf'


def args():
    argv = sys.argv[sys.argv.index('--') + 1:] if '--' in sys.argv else []
    ap = argparse.ArgumentParser(prog='render_actions.py')
    ap.add_argument('--out', required=True)
    ap.add_argument('--actions', default='')
    ap.add_argument('--labels', default='')
    ap.add_argument('--width', type=int, default=1280)
    ap.add_argument('--height', type=int, default=720)
    ap.add_argument('--samples', type=int, default=16)
    ap.add_argument('--yaw', type=float, default=40.0, help='degrees round from the side, toward the front')
    ap.add_argument('--pitch', type=float, default=14.0, help='degrees down')
    return ap.parse_args(argv)


def bounds(objs):
    deps = bpy.context.evaluated_depsgraph_get()
    lo, hi = Vector((1e9,) * 3), Vector((-1e9,) * 3)
    for o in objs:
        ev = o.evaluated_get(deps)
        verts = ev.data.vertices
        for i in range(0, len(verts), 7):
            w = ev.matrix_world @ verts[i].co
            lo = Vector(map(min, lo, w))
            hi = Vector(map(max, hi, w))
    return lo, hi


def studio(lo, hi, samples):
    scene = bpy.context.scene
    world = bpy.data.worlds.new('v2r_studio')
    world.use_nodes = True
    world.node_tree.nodes['Background'].inputs['Color'].default_value = GREY
    world.node_tree.nodes['Background'].inputs['Strength'].default_value = 0.9
    scene.world = world
    for name, rot, energy in (('v2r_key', (math.radians(50), 0, math.radians(-35)), 3.2),
                              ('v2r_fill', (math.radians(60), 0, math.radians(140)), 1.2)):
        light = bpy.data.lights.new(name, 'SUN')
        light.energy = energy
        light.angle = math.radians(12)
        ob = bpy.data.objects.new(name, light)
        ob.rotation_euler = rot
        scene.collection.objects.link(ob)
    r = (hi - lo).length * 20
    floor = bpy.data.meshes.new('v2r_floor')
    floor.from_pydata([(-r, -r, lo.z), (r, -r, lo.z), (r, r, lo.z), (-r, r, lo.z)], [], [(0, 1, 2, 3)])
    mat = bpy.data.materials.new('v2r_floor')
    mat.use_nodes = True
    bsdf = mat.node_tree.nodes['Principled BSDF']
    bsdf.inputs['Base Color'].default_value = GREY
    bsdf.inputs['Roughness'].default_value = 0.9
    floor.materials.append(mat)
    scene.collection.objects.link(bpy.data.objects.new('v2r_floor', floor))
    try:
        scene.render.engine = 'BLENDER_EEVEE'
    except TypeError:
        scene.render.engine = 'BLENDER_EEVEE_NEXT'
    scene.eevee.taa_render_samples = samples
    scene.view_settings.view_transform = 'AgX'
    scene.render.image_settings.file_format = 'PNG'
    scene.render.resolution_percentage = 100


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


def facing(arm):
    v = Vector((0, 0, 0))
    for pb in arm.pose.bones:
        if any(s in pb.name.lower() for s in ('foot', 'toe', 'paw', 'hoof')):
            d = (arm.matrix_world @ pb.tail) - (arm.matrix_world @ pb.head)
            v += Vector((d.x, d.y, 0))
    return v.normalized() if v.length > 1e-6 else Vector((0, -1, 0))


def main():
    a = args()
    scene = bpy.context.scene
    arm = next(o for o in scene.objects if o.type == 'ARMATURE')
    meshes = [o for o in scene.objects if o.type == 'MESH' and o.visible_get()]
    names = [n for n in a.actions.split(',') if n] or [act.name for act in bpy.data.actions
                                                       if any('pose.bones' in fc.data_path for fc in
                                                              action_fcurves(act))]
    if not names:
        raise SystemExit('render_actions: no action with bone keys in this file (blender/import_motion.py makes them)')
    acts = [bpy.data.actions[n] for n in names]
    labels = [s for s in a.labels.split(',') if s] or names
    if arm.animation_data is None:
        arm.animation_data_create()
    # the space every action moves through, sampled
    lo, hi = Vector((1e9,) * 3), Vector((-1e9,) * 3)
    for act in acts:
        arm.animation_data.action = act
        if hasattr(arm.animation_data, 'action_slot') and arm.animation_data.action_slot is None and len(act.slots):
            arm.animation_data.action_slot = act.slots[0]
        f0, f1 = (int(math.floor(v)) for v in act.frame_range)
        for f in range(f0, f1 + 1, max(1, (f1 - f0) // 8)):
            scene.frame_set(f)
            l, h = bounds(meshes)
            lo, hi = Vector(map(min, lo, l)), Vector(map(max, hi, h))
    studio(lo, hi, a.samples)
    fwd = facing(arm)
    side = Vector((0, 0, 1)).cross(fwd)
    yaw, pitch = math.radians(a.yaw), math.radians(a.pitch)
    look = (-side * math.cos(yaw) - fwd * math.sin(yaw))
    look = (look * math.cos(pitch) - Vector((0, 0, 1)) * math.sin(pitch)).normalized()
    centre = (lo + hi) / 2
    size = (hi - lo).length
    cam_data = bpy.data.cameras.new('v2r_film')
    cam_data.lens = 50
    cam = bpy.data.objects.new('v2r_film', cam_data)
    scene.collection.objects.link(cam)
    cam.location = centre - look * size * 2.1
    cam.rotation_euler = look.to_track_quat('-Z', 'Y').to_euler()
    scene.camera = cam
    scene.render.resolution_x, scene.render.resolution_y = a.width, a.height
    out = os.path.abspath(a.out)
    frames_dir = os.path.splitext(out)[0] + '_frames'
    shutil.rmtree(frames_dir, ignore_errors=True)
    os.makedirs(frames_dir)
    fps = scene.render.fps / scene.render.fps_base
    parts = []
    for k, (act, label) in enumerate(zip(acts, labels)):
        arm.animation_data.action = act
        if hasattr(arm.animation_data, 'action_slot') and len(act.slots):
            arm.animation_data.action_slot = act.slots[0]
        f0, f1 = (int(math.floor(v)) for v in act.frame_range)
        for n, f in enumerate(range(f0, f1 + 1)):
            scene.frame_set(f)
            scene.render.filepath = os.path.join(frames_dir, f'{k:02d}_{n:04d}.png')
            bpy.ops.render.render(write_still=True)
        part = os.path.join(frames_dir, f'part{k:02d}.mp4')
        text = label.replace(':', r'\:')
        subprocess.run(['ffmpeg', '-v', 'error', '-y', '-framerate', f'{fps:g}', '-i',
                        os.path.join(frames_dir, f'{k:02d}_%04d.png'), '-vf',
                        f"drawtext=fontfile='{FONT}':text='{text}':x=24:y=24:fontsize=34:fontcolor=white:"
                        f"box=1:boxcolor=black@0.45:boxborderw=10",
                        '-c:v', 'libx264', '-pix_fmt', 'yuv420p', '-crf', '18', part], check=True)
        parts.append(part)
        print(f'RENDER_ACTIONS {label}: {f1 - f0 + 1} frames', flush=True)
    listing = os.path.join(frames_dir, 'parts.txt')
    with open(listing, 'w', newline='\n') as fh:
        for p in parts:
            fh.write(f"file '{p}'\n")
    subprocess.run(['ffmpeg', '-v', 'error', '-y', '-f', 'concat', '-safe', '0', '-i', listing, '-c', 'copy', out],
                   check=True)
    print('RENDER_ACTIONS done ->', out, flush=True)


try:
    main()
except Exception:
    traceback.print_exc()
    sys.exit(1)
