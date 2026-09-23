"""
Shared helpers for all Banana Time Blender scripts.

Run scripts with:  blender -b -P tools/blender/<script>.py
Every script should start with:
    import sys; sys.path.insert(0, "/home/user/AF1/tools/blender"); import btcommon as bt

COORDINATES (important):
  Unity is left-handed Y-up, Blender is right-handed Z-up.  With our FBX export settings
  (axis_forward='-Z', axis_up='Y') and Unity's ModelImporter.bakeAxisConversion = true:
      Unity (x, y, z)  ==  Blender (-x, -z, y)
  i.e. Unity +Z (forward, down the road, away from the camera) == Blender -Y
       Unity +X (screen right)                                  == Blender -X
       Unity +Y (up)                                            == Blender +Z
  A character modelled in Blender facing -Y (Blender "front" view) faces Unity +Z.
  Use u2b()/b2u() to convert positions.  1 Blender unit = 1 metre = 1 Unity unit.
"""
import bpy, bmesh, math, os
from mathutils import Vector, Matrix, Euler

REPO = "/home/user/AF1"
ASSETS = REPO + "/Assets/BananaTime"
MODELS = ASSETS + "/Models"
TEXTURES = ASSETS + "/Art/Textures"
PREVIEWS = REPO + "/tools/previews"

# ---------------------------------------------------------------- game layout (Unity units)
ROAD_HALF_WIDTH = 3.0          # walkable road spans x in [-3, 3], surface at y = 0
WALL_INNER_X = 3.0             # inner face of the side walls
WALL_THICKNESS = 0.7
WALL_HEIGHT = 0.9
SEGMENT_LENGTH = 12.0          # each moving road segment spans z in [0, 12] in its local space
GATE_Z = 72.0                  # temple gate (end of the road) world z; doorway plane at z = 72
MONKEY_Z = 0.0
CAMERA_POS = (0.0, 8.0, -4.6)  # Unity world position of the gameplay camera
CAMERA_PITCH = 31.0            # degrees, looking down (rotation x in Unity)
CAMERA_HFOV = 56.0             # horizontal FOV kept constant in portrait (degrees)
CAMERA_VFOV_MIN, CAMERA_VFOV_MAX = 72.0, 100.0   # portrait 9:16 -> 86.7 deg vertical
MONKEY_HEIGHT = 1.5
LANES = (-2.4, -1.2, 0.0, 1.2, 2.4)
MONKEY_X_LIMIT = 2.4
BANANA_LENGTH = 0.9
COCONUT_DIAMETER = 0.75
COIN_DIAMETER = 1.0

# Sun (direction TOWARDS the sun, Unity world space) – shared with the Unity shader defaults
SUN_DIR_UNITY = (-0.45, 0.80, -0.40)
SUN_COLOR = (1.0, 0.93, 0.80)
SKY_AMBIENT = (0.56, 0.70, 0.86)
GROUND_AMBIENT = (0.42, 0.38, 0.30)


def u2b(x, y, z):
    return Vector((-x, -z, y))


def b2u(v):
    return (-v[0], v[2], -v[1])


def vfov_for_aspect(aspect_w_over_h, hfov=CAMERA_HFOV):
    v = 2.0 * math.degrees(math.atan(math.tan(math.radians(hfov) * 0.5) / aspect_w_over_h))
    return max(CAMERA_VFOV_MIN, min(CAMERA_VFOV_MAX, v))


# ---------------------------------------------------------------- scene helpers
def clear_scene():
    bpy.ops.wm.read_factory_settings(use_empty=True)
    for c in (bpy.data.meshes, bpy.data.materials, bpy.data.images, bpy.data.objects,
              bpy.data.armatures, bpy.data.actions, bpy.data.cameras, bpy.data.lights):
        for b in list(c):
            c.remove(b)


def link(obj, collection=None):
    (collection or bpy.context.scene.collection).objects.link(obj)
    return obj


def new_mesh_object(name, bm_or_mesh):
    if isinstance(bm_or_mesh, bmesh.types.BMesh):
        me = bpy.data.meshes.new(name)
        bm_or_mesh.to_mesh(me)
        bm_or_mesh.free()
    else:
        me = bm_or_mesh
    ob = bpy.data.objects.new(name, me)
    link(ob)
    return ob


def ensure_color_attribute(ob, name="Col", default=(1, 1, 1, 1)):
    """Every exported mesh MUST carry a vertex colour layer (white if unused)."""
    me = ob.data
    if name not in me.color_attributes:
        attr = me.color_attributes.new(name=name, type='BYTE_COLOR', domain='CORNER')
        for d in attr.data:
            d.color = default
    me.color_attributes.active_color = me.color_attributes[name]
    return me.color_attributes[name]


def preview_material(name, color=(0.8, 0.8, 0.8, 1), texture_path=None, emission=None,
                     emission_strength=1.0, alpha_clip=False, roughness=0.7, use_vertex_color=True,
                     tiling=(1, 1)):
    """Blender material used for preview renders AND whose NAME is exported to FBX.
    The Unity builder remaps materials by name (see MaterialDefs json) – the node setup here
    only matters for Blender previews."""
    m = bpy.data.materials.get(name) or bpy.data.materials.new(name)
    m.use_nodes = True
    nt = m.node_tree
    nt.nodes.clear()
    out = nt.nodes.new("ShaderNodeOutputMaterial")
    bsdf = nt.nodes.new("ShaderNodeBsdfPrincipled")
    nt.links.new(bsdf.outputs[0], out.inputs[0])
    bsdf.inputs["Roughness"].default_value = roughness
    base = None
    if texture_path and os.path.exists(texture_path):
        tex = nt.nodes.new("ShaderNodeTexImage")
        tex.image = bpy.data.images.load(texture_path, check_existing=True)
        if tiling != (1, 1):
            mapn = nt.nodes.new("ShaderNodeMapping")
            coord = nt.nodes.new("ShaderNodeTexCoord")
            mapn.inputs["Scale"].default_value = (tiling[0], tiling[1], 1)
            nt.links.new(coord.outputs["UV"], mapn.inputs[0])
            nt.links.new(mapn.outputs[0], tex.inputs[0])
        mix = nt.nodes.new("ShaderNodeMix")
        mix.data_type = 'RGBA'
        mix.blend_type = 'MULTIPLY'
        mix.inputs[0].default_value = 1.0
        nt.links.new(tex.outputs["Color"], mix.inputs[6])
        mix.inputs[7].default_value = color
        base = mix.outputs[2]
        if alpha_clip:
            nt.links.new(tex.outputs["Alpha"], bsdf.inputs["Alpha"])
    else:
        bsdf.inputs["Base Color"].default_value = color
    if use_vertex_color:
        vc = nt.nodes.new("ShaderNodeVertexColor")
        vc.layer_name = "Col"
        mix2 = nt.nodes.new("ShaderNodeMix")
        mix2.data_type = 'RGBA'
        mix2.blend_type = 'MULTIPLY'
        mix2.inputs[0].default_value = 1.0
        if base is not None:
            nt.links.new(base, mix2.inputs[6])
        else:
            mix2.inputs[6].default_value = color
        nt.links.new(vc.outputs["Color"], mix2.inputs[7])
        base = mix2.outputs[2]
    if base is not None:
        nt.links.new(base, bsdf.inputs["Base Color"])
    if emission is not None:
        bsdf.inputs["Emission Color"].default_value = (*emission[:3], 1)
        bsdf.inputs["Emission Strength"].default_value = emission_strength
    return m


# ---------------------------------------------------------------- camera / light / render
def setup_game_camera(res=(540, 960), name="GameCamera"):
    scn = bpy.context.scene
    cam_data = bpy.data.cameras.new(name)
    cam = bpy.data.objects.new(name, cam_data)
    link(cam)
    cam.location = u2b(*CAMERA_POS)
    cam.rotation_euler = Euler((math.radians(90 - CAMERA_PITCH), 0, math.radians(180)), 'XYZ')
    cam_data.lens_unit = 'FOV'
    cam_data.sensor_fit = 'VERTICAL'
    cam_data.angle_y = math.radians(vfov_for_aspect(res[0] / res[1]))
    cam_data.clip_start = 0.3
    cam_data.clip_end = 1500
    scn.camera = cam
    scn.render.resolution_x, scn.render.resolution_y = res
    scn.render.resolution_percentage = 100
    return cam


def setup_preview_lighting(strength=3.2, world_strength=0.9):
    scn = bpy.context.scene
    ld = bpy.data.lights.new("Sun", 'SUN')
    ld.energy = strength
    ld.color = SUN_COLOR
    ld.angle = math.radians(3)
    sun = bpy.data.objects.new("Sun", ld)
    link(sun)
    d = u2b(*SUN_DIR_UNITY).normalized()
    # sun points along its local -Z; we want it to shine FROM direction d
    sun.rotation_euler = (-d).to_track_quat('-Z', 'Y').to_euler()
    if scn.world is None:
        scn.world = bpy.data.worlds.new("World")
    w = scn.world
    w.use_nodes = True
    bg = w.node_tree.nodes.get("Background")
    bg.inputs[0].default_value = (*SKY_AMBIENT, 1)
    bg.inputs[1].default_value = world_strength
    return sun


def render(path, samples=24, engine='CYCLES', threads=2):
    """Cycles CPU render (Eevee does not work headless here). Keep previews small (<= 720 px)
    and threads low: several agents share 4 CPU cores."""
    scn = bpy.context.scene
    scn.render.engine = engine
    scn.render.threads_mode = 'FIXED'
    scn.render.threads = threads
    if engine == 'CYCLES':
        scn.cycles.samples = samples
        scn.cycles.device = 'CPU'
        scn.cycles.use_denoising = True
        try:
            scn.cycles.denoiser = 'OPENIMAGEDENOISE'
        except Exception:
            pass
    scn.view_settings.view_transform = 'Standard'
    scn.render.image_settings.file_format = 'PNG'
    os.makedirs(os.path.dirname(path), exist_ok=True)
    scn.render.filepath = path
    bpy.ops.render.render(write_still=True)
    print("RENDERED", path)


# ---------------------------------------------------------------- export
def export_fbx(path, objects, with_animation=False):
    """Standard FBX export used for EVERY model. Select only `objects` (+ their children)."""
    bpy.ops.object.select_all(action='DESELECT')
    for o in objects:
        o.select_set(True)
        for c in o.children_recursive:
            c.select_set(True)
    bpy.context.view_layer.objects.active = objects[0]
    os.makedirs(os.path.dirname(path), exist_ok=True)
    bpy.ops.export_scene.fbx(
        filepath=path,
        use_selection=True,
        object_types={'ARMATURE', 'MESH', 'EMPTY'},
        apply_unit_scale=True,
        apply_scale_options='FBX_SCALE_ALL',
        axis_forward='-Z', axis_up='Y',
        use_space_transform=True,
        bake_space_transform=False,
        use_mesh_modifiers=True,
        mesh_smooth_type='FACE',
        colors_type='SRGB',
        use_tspace=False,
        add_leaf_bones=False,
        primary_bone_axis='Y', secondary_bone_axis='X',
        armature_nodetype='NULL',
        bake_anim=with_animation,
        bake_anim_use_all_bones=True,
        bake_anim_use_nla_strips=False,
        bake_anim_use_all_actions=with_animation,
        bake_anim_force_startend_keying=True,
        bake_anim_step=1.0,
        bake_anim_simplify_factor=0.0,
        path_mode='STRIP',
        embed_textures=False,
    )
    print("EXPORTED", path)
