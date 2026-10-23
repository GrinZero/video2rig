"""Shared by the Blender scripts that render a creature from its own .blend (render_poses.py, dev/bench/render_gt.py):
the armature checked against the project's rig.json, and base colour images the .blend cannot find (a texture left
on the machine that made the file) taken from the project's rig/texture_<k>.png, the copy export_rig.py saved."""
import os

import bpy
from mathutils import Matrix


def check_rig(arm, info):
    """None when the armature is the one rig.json was exported from (same bones, same rest matrices), else why not."""
    want = {b['name']: Matrix(b['rest']) for b in info['bones']}
    have = {b.name: b.matrix_local for b in arm.data.bones}
    if set(want) != set(have):
        return f'bones differ: {sorted(set(want) ^ set(have))[:8]}'
    off = max(max(abs(x - y) for rw, rh in zip(want[n], have[n]) for x, y in zip(rw, rh)) for n in want)
    return None if off <= 1e-3 else f'rest matrices differ by {off:.4f}'


def base_image(material):
    """The image node export_rig.py takes as the material's base colour, or None."""
    if material is None or not material.use_nodes:
        return None
    bsdf = next((n for n in material.node_tree.nodes if n.type == 'BSDF_PRINCIPLED'), None)
    if bsdf is None or not bsdf.inputs['Base Color'].links:
        return None
    node = bsdf.inputs['Base Color'].links[0].from_node
    while node is not None and node.type != 'TEX_IMAGE' and node.inputs and node.inputs[0].links:
        node = node.inputs[0].links[0].from_node
    return node if node is not None and node.type == 'TEX_IMAGE' else None


def relink_missing(arm, rig_dir, info, log=print):
    """Missing base colour images -> rig/texture_<k>.png (materials counted in export_rig.py's order: the skinned
    meshes' slots, then the props'). Returns the images relinked."""
    order, seen = [], set()
    skinned = [o for o in bpy.data.objects if o.type == 'MESH' and any(
        m.type == 'ARMATURE' and m.object == arm for m in o.modifiers)]
    for obj in skinned:
        for slot in (obj.material_slots or [None]):
            matl = slot.material if slot is not None else None
            key = matl.name if matl is not None else f'_flat_{obj.name}'
            if key not in seen:
                seen.add(key)
                order.append(matl)
    for pr in info.get('props', []):
        obj = bpy.data.objects.get(pr['name'])
        if obj is None:
            continue
        matl = obj.active_material
        key = matl.name if matl is not None else f'_flat_{pr["name"]}'
        if key not in seen:
            seen.add(key)
            order.append(matl)
    mats = info.get('materials', [])
    done = []
    for k, matl in enumerate(order):
        node = base_image(matl)
        if node is None or node.image is None or k >= len(mats) or 'texture' not in mats[k]:
            continue
        img = node.image
        if img.packed_file is not None or (img.filepath and os.path.exists(bpy.path.abspath(img.filepath))):
            continue
        img.filepath = os.path.join(rig_dir, mats[k]['texture'])
        img.reload()
        done.append(img.name)
        log(f'texture {img.name} (missing) <- {img.filepath}')
    return done
