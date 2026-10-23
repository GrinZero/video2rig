"""Export a rigged model for video2rig: the armature, every mesh skinned to it, optional props, and the CURRENT pose
(the pose of the start frame you render for Seedance).

Run in Blender (no window needed):
    blender model.blend --background --python blender/export_rig.py -- --out my_creature/rig [--armature Armature] [--props Log,Stone]

What it writes into --out:
    rig.json      bones (armature-space rest matrices, parents, lengths), the start pose (each bone's basis), the
                  armature's world matrix, the props (world matrix at the start)
    mesh.npz      vertices in armature space (rest shape), triangles, per-corner UVs, the four strongest bone
                  weights per vertex, a material id per triangle; props follow as rigid pieces
    texture_<i>.png  the base colour image of each material (or a flat colour)

Everything is exported in the ARMATURE's own space, so object transforms (a Mixamo FBX at 0.01 scale, a rotated
armature) are fine. Modifiers other than the Armature are not applied: apply them first (mirror, subdivision...).
Props: mesh objects that move as one rigid piece (a log, a stone, a weapon); by name with --props, or every mesh in
a collection called "props". Their position at the start must match the start frame.
"""
import argparse
import json
import os
import sys

import bpy
import numpy as np


def args():
    argv = sys.argv[sys.argv.index('--') + 1:] if '--' in sys.argv else []
    ap = argparse.ArgumentParser(prog='export_rig.py')
    ap.add_argument('--out', required=True)
    ap.add_argument('--armature', default='')
    ap.add_argument('--props', default='', help='comma-separated object names (else the "props" collection)')
    return ap.parse_args(argv)


def m4(m):
    return [list(map(float, r)) for r in m]


def mesh_arrays(obj, to_space):
    """Rest-shape vertices (no modifiers) in to_space, triangles, per-corner UVs, material per triangle."""
    me = obj.data
    me.calc_loop_triangles()
    n = len(me.vertices)
    co = np.empty(n * 3)
    me.vertices.foreach_get('co', co)
    M = np.array(to_space @ obj.matrix_world)
    co = co.reshape(-1, 3) @ M[:3, :3].T + M[:3, 3]
    nt = len(me.loop_triangles)
    tris = np.empty(nt * 3, np.int64)
    me.loop_triangles.foreach_get('vertices', tris)
    loops = np.empty(nt * 3, np.int64)
    me.loop_triangles.foreach_get('loops', loops)
    if me.uv_layers.active is not None:
        uv = np.empty(len(me.loops) * 2)
        me.uv_layers.active.data.foreach_get('uv', uv)
        tri_uv = uv.reshape(-1, 2)[loops].reshape(-1, 3, 2)
    else:
        tri_uv = np.zeros((nt, 3, 2))
    mat = np.empty(nt, np.int64)
    me.loop_triangles.foreach_get('material_index', mat)
    return co, tris.reshape(-1, 3), tri_uv, mat


def base_colour(material, out_dir, k):
    """Save the material's base colour image as texture_<k>.png; a flat colour when there is none."""
    path = os.path.join(out_dir, f'texture_{k}.png')
    colour = (0.6, 0.6, 0.6)
    if material is not None and material.use_nodes:
        bsdf = next((n for n in material.node_tree.nodes if n.type == 'BSDF_PRINCIPLED'), None)
        if bsdf is not None:
            inp = bsdf.inputs['Base Color']
            colour = tuple(inp.default_value[:3])
            link = inp.links[0] if inp.links else None
            node = link.from_node if link else None
            while node is not None and node.type != 'TEX_IMAGE' and node.inputs and node.inputs[0].links:
                node = node.inputs[0].links[0].from_node          # through mix / colour ramp nodes
            if node is not None and node.type == 'TEX_IMAGE' and node.image is not None and node.image.size[0] > 0:
                # copy the pixels into a new image: works for packed, external and generated images alike
                src = node.image
                w, h = src.size
                px = np.empty(w * h * 4, np.float32)
                src.pixels.foreach_get(px)
                img = bpy.data.images.new('video2rig_texture', w, h, alpha=True)
                img.pixels.foreach_set(px)
                img.filepath_raw = path
                img.file_format = 'PNG'
                img.save()
                bpy.data.images.remove(img)
                return {'texture': os.path.basename(path)}
    return {'colour': [float(c) for c in colour]}


def main():
    a = args()
    out = os.path.abspath(a.out)
    os.makedirs(out, exist_ok=True)
    arms = [o for o in bpy.data.objects if o.type == 'ARMATURE']
    arm = bpy.data.objects[a.armature] if a.armature else (arms[0] if len(arms) == 1 else None)
    if arm is None:
        sys.exit('export_rig: more than one armature, name it with --armature: ' + ', '.join(o.name for o in arms))
    to_arm = arm.matrix_world.inverted()
    # bones, parents first
    order, seen = [], set()

    def visit(b):
        if b.name in seen:
            return
        if b.parent is not None:
            visit(b.parent)
        seen.add(b.name)
        order.append(b)
    for b in arm.data.bones:
        visit(b)
    bones = [{'name': b.name, 'parent': b.parent.name if b.parent else None, 'rest': m4(b.matrix_local),
              'length': float(b.length), 'deform': bool(b.use_deform)} for b in order]
    names = [b['name'] for b in bones]
    start = {}
    for pb in arm.pose.bones:
        start[pb.name] = m4(pb.matrix_basis)
    # skinned meshes
    skinned = [o for o in bpy.data.objects if o.type == 'MESH' and any(
        m.type == 'ARMATURE' and m.object == arm for m in o.modifiers)]
    if not skinned:
        sys.exit(f'export_rig: no mesh has an Armature modifier pointing at {arm.name}')
    prop_names = [p for p in a.props.split(',') if p]
    if not prop_names and 'props' in bpy.data.collections:
        prop_names = [o.name for o in bpy.data.collections['props'].all_objects if o.type == 'MESH']
    parts = []          # (coords, tris, tri_uv, material ids, idx, wts)
    materials, mat_slot = [], {}
    for obj in skinned:
        co, tris, tri_uv, mat = mesh_arrays(obj, to_arm)
        gi = {g.index: g.name for g in obj.vertex_groups}
        idx = np.zeros((len(co), 4), np.int64)
        wts = np.zeros((len(co), 4))
        for v in obj.data.vertices:
            pairs = sorted(((g.weight, names.index(gi[g.group])) for g in v.groups
                            if gi.get(g.group) in names and g.weight > 0), reverse=True)[:4]
            s = sum(w for w, _ in pairs)
            for k, (w, j) in enumerate(pairs):
                idx[v.index, k], wts[v.index, k] = j, w / s
        unweighted = int((wts.sum(1) == 0).sum())
        if unweighted:
            print(f'EXPORT_RIG warning: {obj.name}: {unweighted} vertices without bone weights follow the root bone')
            idx[wts.sum(1) == 0, 0] = 0
            wts[wts.sum(1) == 0, 0] = 1.0
        ids = []
        for slot in (obj.material_slots or [None]):
            matl = slot.material if slot is not None else None
            key = matl.name if matl is not None else f'_flat_{obj.name}'
            if key not in mat_slot:
                mat_slot[key] = len(materials)
                materials.append(base_colour(matl, out, len(materials)))
            ids.append(mat_slot[key])
        parts.append((co, tris, tri_uv, np.array(ids)[np.clip(mat, 0, len(ids) - 1)], idx, wts))
    props = []
    for pi, name in enumerate(prop_names):
        obj = bpy.data.objects.get(name)
        if obj is None or obj.type != 'MESH':
            sys.exit(f'export_rig: prop {name} is not a mesh object')
        co, tris, tri_uv, mat = mesh_arrays(obj, to_arm)
        key = obj.active_material.name if obj.active_material else f'_flat_{name}'
        if key not in mat_slot:
            mat_slot[key] = len(materials)
            materials.append(base_colour(obj.active_material, out, len(materials)))
        idx = np.full((len(co), 4), len(names) + pi, np.int64)       # a prop is a pseudo bone after the real ones
        wts = np.zeros((len(co), 4)); wts[:, 0] = 1.0
        parts.append((co, tris, tri_uv, np.full(len(tris), mat_slot[key]), idx, wts))
        centre = co.mean(0)
        props.append({'name': name, 'centre': [float(c) for c in centre], 'vertices': int(len(co))})
    offset, all_co, all_tris, all_uv, all_mat, all_idx, all_wts = 0, [], [], [], [], [], []
    for co, tris, tri_uv, mat, idx, wts in parts:
        all_co.append(co); all_tris.append(tris + offset); all_uv.append(tri_uv); all_mat.append(mat)
        all_idx.append(idx); all_wts.append(wts)
        offset += len(co)
    np.savez_compressed(os.path.join(out, 'mesh.npz'), verts=np.vstack(all_co), tris=np.vstack(all_tris),
                        tri_uv=np.vstack(all_uv), tri_material=np.concatenate(all_mat), idx=np.vstack(all_idx),
                        wts=np.vstack(all_wts))
    info = {'source': os.path.basename(bpy.data.filepath), 'armature': arm.name,
            'armature_world': m4(arm.matrix_world), 'bones': bones, 'start_pose': start,
            'props': props, 'materials': materials, 'meshes': [o.name for o in skinned]}
    with open(os.path.join(out, 'rig.json'), 'w', newline='\n') as f:
        json.dump(info, f, indent=1)
        f.write('\n')
    print(f'EXPORT_RIG {len(bones)} bones, {offset} vertices, {sum(len(t) for t in all_tris)} triangles, '
          f'{len(props)} props, {len(materials)} materials -> {out}', flush=True)


main()
