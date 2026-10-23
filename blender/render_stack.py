"""The STACKED start frame for Seedance: the side view above, the view from directly above below it, both 1920 px
wide, at ONE scale and lined up, so every point of the body sits in the same column in both strips (the forward axis
runs left in both). Side|top panels side by side drew forward motion horizontally in one panel and vertically in the
other, and Seedance let the two drift apart in fast travelling moves (a leap: up to 1.6 m in the air); stacked, both
strips share the forward axis in the same image direction, as front|side panels share the height (one image row):
the same leap agreed to about 0.1 m in the air (yt/ep-boss-fight/v2r/RUN.md, 2026-10-18).

    blender model.blend --background --python blender/render_stack.py -- --out my_creature/start
        [--facing +X] [--side-margin 1.93] [--top-margin 1.8] [--x-at 0.62] [--divider 4] [--samples 64]
        [--read-like other_start/cameras.json] [--cameras-only]

--side-margin: the side strip's height / the body's height (room to rear up and leap); --top-margin: the top strip's
height / the body's width (room to sway); the scale follows from both strips sharing 1080 px. --x-at: the body's
middle at this share of the width from the left (room ahead: forward is the image's left). --read-like: the
cameras.json of a side|top start frame whose clips trained the reader: each strip gets a reading window
(clip.Panel) that shows the body as big as that frame's panel of the same name did (a 1920 px strip letterboxed whole
would show it at half that size). --cameras-only: write cameras.json again without rendering (same model, same
options: the same cameras).

cameras.json: the side strip is a plain panel with y0_px 0; the strip from above lies in the frame under it (y0_px)
and is described upright, as the side|top frame's view from above (head up): turn 3 (np.rot90's k) cuts it from the
frame and turns it so.
"""
import argparse
import json
import math
import os
import sys

import bpy
import numpy as np
from mathutils import Matrix, Vector

GREY = (0.42, 0.41, 0.40, 1.0)
W, H = 1920, 1080


def args():
    argv = sys.argv[sys.argv.index('--') + 1:] if '--' in sys.argv else []
    ap = argparse.ArgumentParser(prog='render_stack.py')
    ap.add_argument('--out', required=True)
    ap.add_argument('--facing', default='+X')
    ap.add_argument('--side-margin', type=float, default=1.93)
    ap.add_argument('--top-margin', type=float, default=1.8)
    ap.add_argument('--x-at', type=float, default=0.62)
    ap.add_argument('--divider', type=int, default=4)
    ap.add_argument('--samples', type=int, default=64)
    ap.add_argument('--read-like', default='')
    ap.add_argument('--cameras-only', action='store_true')
    return ap.parse_args(argv)


def bounds(objs):
    deps = bpy.context.evaluated_depsgraph_get()
    lo, hi = Vector((1e9,) * 3), Vector((-1e9,) * 3)
    for o in objs:
        ev = o.evaluated_get(deps)
        for v in ev.data.vertices:
            w = ev.matrix_world @ v.co
            lo = Vector(map(min, lo, w))
            hi = Vector(map(max, hi, w))
    return lo, hi


def studio(lo, hi, samples):
    """render_start.py's studio: grey world, a key and a fill sun, a grey floor."""
    scene = bpy.context.scene
    world = bpy.data.worlds.new('video2rig_studio')
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
    r = (hi - lo).length * 10
    floor = bpy.data.meshes.new('v2r_floor')
    floor.from_pydata([(-r, -r, lo.z), (r, -r, lo.z), (r, r, lo.z), (-r, r, lo.z)], [], [(0, 1, 2, 3)])
    mat = bpy.data.materials.new('v2r_floor')
    mat.use_nodes = True
    bsdf = mat.node_tree.nodes['Principled BSDF']
    bsdf.inputs['Base Color'].default_value = GREY
    bsdf.inputs['Roughness'].default_value = 0.9
    floor.materials.append(mat)
    scene.collection.objects.link(bpy.data.objects.new('v2r_floor', floor))
    scene.render.engine = 'CYCLES'
    scene.cycles.device = 'CPU'
    scene.cycles.samples = samples
    scene.cycles.use_denoising = True
    scene.view_settings.view_transform = 'AgX'


def camera(name, look, target, ortho_h, w, h, up):
    """An orthographic camera looking along `look`, its image's up along `up`, ortho_h metres tall."""
    data = bpy.data.cameras.new(name)
    data.type = 'ORTHO'
    data.sensor_fit = 'VERTICAL'
    data.ortho_scale = ortho_h
    cam = bpy.data.objects.new(name, data)
    bpy.context.scene.collection.objects.link(cam)
    cam.location = target - look * 30.0
    z = -look.normalized()
    y = (up - z * up.dot(z)).normalized()
    x = y.cross(z)
    cam.rotation_euler = Matrix((x, y, z)).transposed().to_euler()
    bpy.context.view_layer.update()
    R = cam.matrix_world.to_3x3().normalized()
    return cam, {'name': name, 'width_px': w, 'height_px': h, 'type': 'ortho', 'location': list(cam.location),
                 'right': list(R.col[0]), 'up': list(R.col[1]), 'forward': list(-R.col[2]),
                 'ortho_width': ortho_h * w / h, 'ortho_height': ortho_h}


def main():
    a = args()
    out = os.path.abspath(a.out)
    os.makedirs(out, exist_ok=True)
    scene = bpy.context.scene
    objs = [o for o in scene.objects if o.type == 'MESH' and o.visible_get()]
    lo, hi = bounds(objs)
    studio(lo, hi, a.samples)
    fwd = Vector({'+X': (1, 0, 0), '-X': (-1, 0, 0), '+Y': (0, 1, 0), '-Y': (0, -1, 0)}[a.facing])
    left = Vector((0, 0, 1)).cross(fwd)
    body_h = hi.z - lo.z
    corners = [Vector((x, y, 0)) for x in (lo.x, hi.x) for y in (lo.y, hi.y)]
    length = max(c.dot(fwd) for c in corners) - min(c.dot(fwd) for c in corners)
    width = max(c.dot(left) for c in corners) - min(c.dot(left) for c in corners)
    side_m, top_m = body_h * a.side_margin, width * a.top_margin
    k = (H - a.divider) / (side_m + top_m)                         # px per metre, both strips
    h_side = int(round(side_m * k))
    h_top = H - a.divider - h_side
    side_m, top_m = h_side / k, h_top / k
    centre = (lo + hi) * .5
    c = centre + fwd * ((a.x_at - .5) * W / k)                     # forward runs to the image's left in both strips
    side_target = Vector((c.x, c.y, lo.z + side_m * (.5 - .12)))   # the floor line 12 % above the strip's bottom
    top_target = Vector((c.x, c.y, centre.z))
    side_cam, side = camera('v2r_side', -left, side_target, side_m, W, h_side, Vector((0, 0, 1)))
    top_cam, top = camera('v2r_top', Vector((0, 0, -1)), top_target, top_m, W, h_top, -left)
    if not a.cameras_only:
        paths = []
        for cam, rec, h in ((side_cam, side, h_side), (top_cam, top, h_top)):
            scene.camera = cam
            scene.render.resolution_x, scene.render.resolution_y = W, h
            scene.render.resolution_percentage = 100
            path = os.path.join(out, f'panel-{rec["name"][4:]}.png')
            scene.render.filepath = path
            bpy.ops.render.render(write_still=True)
            paths.append(path)
            print('RENDER_STACK', rec['name'], W, h, '->', path, flush=True)
        strips = []
        for p in paths:
            img = bpy.data.images.load(p)
            strips.append(np.array(img.pixels[:], np.float32).reshape(img.size[1], img.size[0], 4))
            bpy.data.images.remove(img)
        div = np.zeros((a.divider, W, 4), np.float32)
        div[..., :3], div[..., 3] = .03, 1.0
        full = np.concatenate([strips[1], div, strips[0]], 0)       # Blender's rows run bottom-up: the side strip last
        img = bpy.data.images.new('v2r_stack', W, H, alpha=True)
        img.pixels.foreach_set(full.ravel())
        img.filepath_raw = os.path.join(out, 'start.png')
        img.file_format = 'PNG'
        img.save()
    # the strip from above, described upright (turned a quarter clockwise: np.rot90 k = 3): its right is the
    # rendered image's up, its up the rendered image's left
    up_top = [-x for x in top['right']]
    top = dict(top, right=top['up'], up=up_top, width_px=h_top, height_px=W, ortho_width=top['ortho_height'],
               ortho_height=top['ortho_width'], turn=3)
    side.update(x0_px=0, y0_px=0)
    top.update(x0_px=0, y0_px=h_side + a.divider)
    if a.read_like:
        ref = {v['name']: v for v in json.loads(open(a.read_like, encoding='utf-8').read())['views']}
        for rec in (side, top):
            r = ref.get(rec['name'])
            if r is None:
                continue
            k_ref = r['width_px'] / r['ortho_width']
            rec['read_window'] = [round(r['width_px'] * k / k_ref), round(r['height_px'] * k / k_ref)]
    with open(os.path.join(out, 'cameras.json'), 'w', newline='\n') as f:
        json.dump({'frame_px': [W, H], 'layout': 'stacked', 'views': [side, top], 'hover': 0.0}, f, indent=1)
        f.write('\n')
    print(f'RENDER_STACK done: {k:.1f} px/m, side {W}x{h_side} ({side_m:.2f} m tall, {W / k:.1f} m wide), top '
          f'{W}x{h_top} ({top_m:.2f} m), body {body_h:.2f} x {length:.2f} x {width:.2f} m, windows '
          f'{side.get("read_window")} / {top.get("read_window")} -> {out}', flush=True)


main()
