"""Write cameras.json for a start frame you rendered yourself (render_start.py writes it on its own).

Run in Blender with the scene that rendered the start frame:
    blender scene.blend --background --python blender/export_camera.py -- --out my_creature/start --cameras Camera
    blender scene.blend --background --python blender/export_camera.py -- --out my_creature/start --cameras CamFront,CamSide

One camera = the whole frame. Several = panels side by side, left to right in the order given (a split start
frame, each panel rendered at the scene resolution and joined horizontally). Orthographic and perspective cameras.
Panels of different widths (a wide side view next to a narrow top view, both at the scene's height): give each
panel's width in pixels in the same order, --widths 1280,640.
"""
import argparse
import json
import os
import sys

import bpy


def camera_record(cam, w, h, x0):
    M = cam.matrix_world
    R = M.to_3x3().normalized()
    rec = {'name': cam.name, 'x0_px': x0, 'width_px': w, 'height_px': h,
           'location': list(M.translation), 'right': list(R.col[0]), 'up': list(R.col[1]),
           'forward': list(-R.col[2])}
    d = cam.data
    sensor_fit = d.sensor_fit if d.sensor_fit != 'AUTO' else ('HORIZONTAL' if w >= h else 'VERTICAL')
    if d.type == 'ORTHO':
        if sensor_fit == 'HORIZONTAL':
            ow, oh = d.ortho_scale, d.ortho_scale * h / w
        else:
            ow, oh = d.ortho_scale * w / h, d.ortho_scale
        rec.update(type='ortho', ortho_width=ow, ortho_height=oh)
    elif d.type == 'PERSP':
        sensor = d.sensor_width if sensor_fit == 'HORIZONTAL' else d.sensor_height
        size = w if sensor_fit == 'HORIZONTAL' else h
        rec.update(type='persp', focal_px=d.lens / sensor * size)
    else:
        sys.exit(f'export_camera: {cam.name} is {d.type}; use an orthographic or perspective camera')
    if abs(d.shift_x) > 1e-6 or abs(d.shift_y) > 1e-6:
        s = max(w, h)
        rec['shift_px'] = [d.shift_x * s, -d.shift_y * s]
    return rec


def main():
    argv = sys.argv[sys.argv.index('--') + 1:] if '--' in sys.argv else []
    ap = argparse.ArgumentParser(prog='export_camera.py')
    ap.add_argument('--out', required=True)
    ap.add_argument('--cameras', default='')
    ap.add_argument('--widths', default='')
    a = ap.parse_args(argv)
    scene = bpy.context.scene
    r = scene.render
    w = int(r.resolution_x * r.resolution_percentage / 100)
    h = int(r.resolution_y * r.resolution_percentage / 100)
    names = [c for c in a.cameras.split(',') if c] or [scene.camera.name]
    widths = [int(v) for v in a.widths.split(',') if v] or [w] * len(names)
    if len(widths) != len(names):
        sys.exit(f'export_camera: {len(names)} cameras but {len(widths)} widths')
    views, x0 = [], 0
    for name, wk in zip(names, widths):
        cam = bpy.data.objects.get(name)
        if cam is None or cam.type != 'CAMERA':
            sys.exit(f'export_camera: no camera called {name}')
        views.append(camera_record(cam, wk, h, x0))
        x0 += wk
    os.makedirs(a.out, exist_ok=True)
    data = {'frame_px': [x0, h], 'views': views}
    with open(os.path.join(a.out, 'cameras.json'), 'w', newline='\n') as f:
        json.dump(data, f, indent=1)
        f.write('\n')
    print('EXPORT_CAMERA', ', '.join(v['name'] for v in views), '->', a.out, flush=True)


if __name__ == '__main__':
    main()
