"""Auto-rig with UniRig, the Blender side (video2rig.autorig.unirig calls it).

    blender -b --factory-startup <model.blend> --python blender/unirig_io.py -- export <out.glb>
        the model's meshes as a GLB, the input UniRig reads
    blender -b --factory-startup --python blender/unirig_io.py -- read <out.json> <skeleton.fbx> ...
        each UniRig skeleton: its bones (name, parent, head, tail) and the bounds of the mesh copy it was drawn on
        (UniRig scales the model into a box of its own; the bounds map it back)
"""
import json
import sys
import traceback

import bpy


def export(path):
    bpy.ops.object.select_all(action='DESELECT')
    meshes = [o for o in bpy.context.scene.objects if o.type == 'MESH']
    for o in meshes:
        o.select_set(True)
    bpy.context.view_layer.objects.active = meshes[0]
    bpy.ops.export_scene.gltf(filepath=path, export_format='GLB', use_selection=True, export_materials='NONE',
                              export_animations=False, export_skins=False, export_morph=False)


def read(out, paths):
    res = []
    for p in paths:
        bpy.ops.wm.read_factory_settings(use_empty=True)
        bpy.ops.import_scene.fbx(filepath=p)
        arm = next(o for o in bpy.data.objects if o.type == 'ARMATURE')
        lo, hi = [1e9] * 3, [-1e9] * 3
        for o in bpy.data.objects:
            if o.type != 'MESH':
                continue
            for v in o.data.vertices:
                w = o.matrix_world @ v.co
                lo = [min(a, b) for a, b in zip(lo, w)]
                hi = [max(a, b) for a, b in zip(hi, w)]
        M = arm.matrix_world
        bones = [{'name': b.name, 'parent': b.parent.name if b.parent else None,
                  'head': list(M @ b.head_local), 'tail': list(M @ b.tail_local)} for b in arm.data.bones]
        res.append({'path': p, 'lo': lo, 'hi': hi, 'bones': bones})
    with open(out, 'w') as f:
        json.dump(res, f)


try:
    argv = sys.argv[sys.argv.index('--') + 1:]
    if argv[0] == 'export':
        export(argv[1])
    else:
        read(argv[1], argv[2:])
    print('UNIRIG_IO_OK', flush=True)
except Exception:
    traceback.print_exc()
    sys.exit(1)
