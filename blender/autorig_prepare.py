"""Auto-rig, step 1 (Blender): a watertight copy of the model for the skeleton search.

    blender --background --factory-startup --python blender/autorig_prepare.py -- --model creature.glb --out P/autorig
        [--voxel 0.012] [--facing -Y]

Imports the model (.glb/.gltf/.fbx/.obj, or opens a .blend), applies its transforms, saves it as
<out>/model.blend, and writes <out>/shell.npz: the voxel remesh of all its meshes joined (Blender's Remesh closes
every gap and fur card at the voxel size, so inside and outside are defined), plus the model's bounds and the
facing the creature is assumed to have (Tripo models face -Y in Blender; --facing overrides).
"""
import argparse
import os
import sys
import traceback

import bpy
import numpy as np


def args():
    argv = sys.argv[sys.argv.index('--') + 1:] if '--' in sys.argv else []
    ap = argparse.ArgumentParser(prog='autorig_prepare.py')
    ap.add_argument('--model', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--voxel', type=float, default=0.012, help='remesh voxel, a share of the model height')
    ap.add_argument('--facing', default='-Y')
    return ap.parse_args(argv)


def load(path):
    ext = os.path.splitext(path)[1].lower()
    if ext == '.blend':
        bpy.ops.wm.open_mainfile(filepath=path)
        return
    bpy.ops.wm.read_factory_settings(use_empty=True)
    if ext in ('.glb', '.gltf'):
        bpy.ops.import_scene.gltf(filepath=path)
    elif ext == '.fbx':
        bpy.ops.import_scene.fbx(filepath=path)
    elif ext == '.obj':
        bpy.ops.wm.obj_import(filepath=path)
    else:
        raise SystemExit(f'autorig: cannot import {ext}')


def main():
    a = args()
    out = os.path.abspath(a.out)
    os.makedirs(out, exist_ok=True)
    load(os.path.abspath(a.model))
    scene = bpy.context.scene
    # no armature, no animation: the raw mesh in its own pose
    for o in list(scene.objects):
        if o.type == 'ARMATURE':
            for m in [c for c in scene.objects if c.parent == o]:
                mw = m.matrix_world.copy()
                m.parent = None
                m.matrix_world = mw
            bpy.data.objects.remove(o)
    meshes = [o for o in scene.objects if o.type == 'MESH']
    for o in meshes:
        for mod in list(o.modifiers):
            if mod.type == 'ARMATURE':
                o.modifiers.remove(mod)
        o.vertex_groups.clear()
    bpy.ops.object.select_all(action='DESELECT')
    for o in meshes:
        o.select_set(True)
    bpy.context.view_layer.objects.active = meshes[0]
    bpy.ops.object.transform_apply(location=True, rotation=True, scale=True)
    # the model on the floor, centred over the origin
    lo = np.full(3, 1e9)
    hi = np.full(3, -1e9)
    for o in meshes:
        V = np.array([o.matrix_world @ v.co for v in o.data.vertices])
        lo, hi = np.minimum(lo, V.min(0)), np.maximum(hi, V.max(0))
    shift = np.array([-(lo[0] + hi[0]) / 2, -(lo[1] + hi[1]) / 2, -lo[2]])
    for o in meshes:
        o.location = tuple(np.array(o.location) + shift)
    bpy.ops.object.transform_apply(location=True, rotation=False, scale=False)
    lo, hi = lo + shift, hi + shift
    height = float(hi[2] - lo[2])
    bpy.ops.wm.save_as_mainfile(filepath=os.path.join(out, 'model.blend'))
    # the watertight shell: a joined copy, voxel-remeshed
    bpy.ops.object.duplicate()
    copies = [o for o in bpy.context.selected_objects]
    bpy.context.view_layer.objects.active = copies[0]
    if len(copies) > 1:
        bpy.ops.object.join()
    shell = bpy.context.view_layer.objects.active

    def triangles(me):
        V = np.array([v.co for v in me.vertices], np.float32)
        F = []
        for p in me.polygons:
            vs = list(p.vertices)
            for k in range(1, len(vs) - 1):
                F.append((vs[0], vs[k], vs[k + 1]))
        return V, np.array(F, np.int32)
    # the model's own skin as it is, beside the shell: a sheet with one side (a robe's strips, a hood, a membrane)
    # has no inside, and the remesh keeps only shreds of it (a wraith's robe: 30-70 voxels a level cut)
    SV, SF = triangles(shell.data)
    mod = shell.modifiers.new('remesh', 'REMESH')
    mod.mode = 'VOXEL'
    mod.voxel_size = a.voxel * height
    mod.use_smooth_shade = False
    bpy.ops.object.modifier_apply(modifier=mod.name)
    V, F = triangles(shell.data)
    np.savez_compressed(os.path.join(out, 'shell.npz'), verts=V, faces=F, lo=lo, hi=hi,
                        height=height, facing=a.facing, voxel=a.voxel * height,
                        meshes=np.array([o.name for o in meshes]), skin_verts=SV, skin_faces=SF)
    print(f'AUTORIG_PREPARE {len(meshes)} meshes, height {height:.3f}, shell {len(V)} verts -> {out}', flush=True)


try:
    main()
except Exception:
    traceback.print_exc()
    sys.exit(1)
