"""A marked start frame: your model as in the start frame, its limbs painted like a motion-capture suit, through the
very cameras of start/cameras.json. Each limb that hangs down from the body (legs, arms; not tails or ears) gets its
own colour, darker above the elbow or knee, full below it, light on the hand or paw, near white on the toes, so the
video model draws a creature whose thin, fast, look-alike limbs stay apart, and every bend shows as the edge between
two colours. The fit and the final animation still use the plain model: the colours only help the video.

Run in Blender with the scene the start frame was rendered from (the same pose):
    blender rig.blend --background --python blender/render_markers.py -- --cameras my_wolf/start/cameras.json --out my_wolf/start-marked
Writes start.png (the panels joined as in cameras.json) and markers.json (which bone got which colour).
"""
import argparse
import json
import math
import os
import sys

import bpy
from mathutils import Matrix, Vector

# the studio of render_start.py (that script runs on import)
GREY = (0.42, 0.41, 0.40, 1.0)


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


def studio(lo, hi, samples, engine):
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
    scene.render.engine = engine
    if engine == 'CYCLES':
        scene.cycles.device = 'CPU'
        scene.cycles.samples = samples
        scene.cycles.use_denoising = True
    scene.view_settings.view_transform = 'AgX'

HUES = [0.0, 0.6, 0.15, 0.33, 0.8, 0.07, 0.5, 0.92]      # red, blue, yellow, green, purple, orange, cyan, pink
SHADES = [.45, 1.0, 1.0, 1.0]                            # value along the limb: above the hinge darker
TINTS = [0.0, 0.0, .45, .8]                              # mixed toward white: hand or paw, then toes
FINGER = ('finger', 'thumb', 'index', 'middle', 'ring', 'pinky', 'eye', 'jaw', 'tongue', 'teeth', 'end')


def args():
    argv = sys.argv[sys.argv.index('--') + 1:] if '--' in sys.argv else []
    ap = argparse.ArgumentParser(prog='render_markers.py')
    ap.add_argument('--cameras', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--samples', type=int, default=64)
    ap.add_argument('--engine', default='CYCLES')
    return ap.parse_args(argv)


def limbs(arm):
    """Chains of bones leaving a hub (a bone with three or more children) without branching, pointing down."""
    bones = arm.data.bones
    kids = {b.name: [c for c in b.children if not any(f in c.name.lower() for f in FINGER)] for b in bones}
    out = []
    for hub in bones:
        if len(kids[hub.name]) < 3:
            continue
        for c in kids[hub.name]:
            chain, b = [], c
            while True:
                chain.append(b)
                if len(kids[b.name]) != 1:
                    break
                b = kids[b.name][0]
            M = arm.matrix_world
            top, end = M @ chain[0].head_local, M @ chain[-1].tail_local
            if len(chain) >= 3 and end.z < top.z - .3 * (top - end).length:
                out.append(chain)
    return out


def colour(h, shade, tint):
    import colorsys
    r, g, b = colorsys.hsv_to_rgb(h, .9, shade)
    return tuple(c + (1 - c) * tint for c in (r, g, b)) + (1.0,)


def paint(arm, meshes):
    """Faces go to the colour of their corners' strongest bone, where that bone is a limb's (from its second bone
    down: the first is mostly body on screen); everything else keeps its own material."""
    chains = limbs(arm)
    names = [c[0].name for c in chains]
    paint_of = {}
    for i, chain in enumerate(chains):
        hue = HUES[i % len(HUES)]
        in_chain = {x.name for x in chain}
        for k, b in enumerate(chain[1:]):
            kk = min(k, len(SHADES) - 1)
            stack = [b]
            while stack:                                   # the bone and what branches off it (fingers, toes)
                x = stack.pop()
                paint_of[x.name] = colour(hue, SHADES[kk], TINTS[kk])
                stack.extend(c for c in x.children if c.name not in in_chain)
    mats = {}
    for name, col in paint_of.items():
        key = tuple(round(c, 3) for c in col)
        if key not in mats:
            m = bpy.data.materials.new(f'marker_{len(mats)}')
            m.use_nodes = True
            bsdf = m.node_tree.nodes['Principled BSDF']
            bsdf.inputs['Base Color'].default_value = col
            bsdf.inputs['Roughness'].default_value = .6
            mats[key] = m
    for ob in meshes:
        groups = {g.index: g.name for g in ob.vertex_groups}
        strongest = []
        for v in ob.data.vertices:
            best = max(v.groups, key=lambda g: g.weight, default=None)
            strongest.append(groups.get(best.group) if best is not None else None)
        slot = {}
        for key, m in mats.items():
            ob.data.materials.append(m)
            slot[key] = len(ob.data.materials) - 1
        for poly in ob.data.polygons:
            votes = [strongest[i] for i in poly.vertices]
            top = max(set(votes), key=votes.count)
            if top in paint_of:
                poly.material_index = slot[tuple(round(c, 3) for c in paint_of[top])]
    return {names[i]: [b.name for b in chain] for i, chain in enumerate(chains)}, \
        {k: [round(c, 3) for c in v[:3]] for k, v in paint_of.items()}


def camera_from(rec):
    data = bpy.data.cameras.new(rec['name'])
    data.type = 'ORTHO'
    data.sensor_fit = 'HORIZONTAL'
    data.ortho_scale = rec['ortho_width']
    cam = bpy.data.objects.new(rec['name'], data)
    bpy.context.scene.collection.objects.link(cam)
    R = Matrix((rec['right'], rec['up'], [-c for c in rec['forward']])).transposed()
    cam.matrix_world = Matrix.Translation(Vector(rec['location'])) @ R.to_4x4()
    return cam


def main():
    a = args()
    out = os.path.abspath(a.out)
    os.makedirs(out, exist_ok=True)
    scene = bpy.context.scene
    arm = next(o for o in scene.objects if o.type == 'ARMATURE')
    meshes = [o for o in scene.objects if o.type == 'MESH' and o.visible_get()
              and any(m.type == 'ARMATURE' for m in o.modifiers)]
    chains, colours = paint(arm, meshes)
    lo, hi = bounds(meshes)
    studio(lo, hi, a.samples, a.engine)
    cams = json.load(open(a.cameras))
    panels = []
    import numpy as np
    for rec in cams['views']:
        cam = camera_from(rec)
        scene.camera = cam
        scene.render.resolution_x, scene.render.resolution_y = rec['width_px'], rec['height_px']
        scene.render.resolution_percentage = 100
        path = os.path.join(out, f'panel-{rec["name"]}.png')
        scene.render.filepath = path
        bpy.ops.render.render(write_still=True)
        img = bpy.data.images.load(path)
        px = np.array(img.pixels[:], np.float32).reshape(img.size[1], img.size[0], 4)
        panels.append(px)
        bpy.data.images.remove(img)
    full = np.concatenate(panels, axis=1)
    x = 0
    for rec in cams['views'][:-1]:                         # the thin dark line between the panels, as in the start frame
        x += rec['width_px']
        full[:, x - 2:x + 1, :3] = .03
    img = bpy.data.images.new('v2r_marked', full.shape[1], full.shape[0], alpha=True)
    img.pixels.foreach_set(full.ravel())
    img.filepath_raw = os.path.join(out, 'start.png')
    img.file_format = 'PNG'
    img.save()
    with open(os.path.join(out, 'markers.json'), 'w', newline='\n') as f:
        json.dump({'limbs': chains, 'colours': colours}, f, indent=1)
    print('RENDER_MARKERS', len(chains), 'limbs ->', out, flush=True)


if __name__ == '__main__':
    main()
