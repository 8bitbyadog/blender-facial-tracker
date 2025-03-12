"""
Face Tracker - Real-time facial motion capture for Blender
This add-on provides tools for real-time face tracking and animation
"""

bl_info = {
    "name": "Face Tracker",
    "author": "8bitbyadog",
    "version": (1, 0, 0),
    "blender": (4, 3, 2),
    "location": "View3D > Sidebar > Face Tracker",
    "description": "Real-time face tracking and animation tools",
    "category": "Animation",
    "support": "COMMUNITY",
    "warning": "",
    "doc_url": "",
    "tracker_url": ""
}

import bpy
import bmesh
import os
import sys
import time
import numpy as np
from pathlib import Path
from threading import Thread
from queue import Queue
import gpu
from gpu_extras.batch import batch_for_shader
import cv2
import traceback

# Add Blender's modules directory to Python path
blender_scripts_path = Path(bpy.utils.user_resource('SCRIPTS'))
modules_path = str(blender_scripts_path / "modules")
if modules_path not in sys.path:
    sys.path.append(modules_path)

try:
    import cv2
    OPENCV_AVAILABLE = True
except ImportError:
    OPENCV_AVAILABLE = False

from bpy.props import StringProperty, BoolProperty, FloatProperty, EnumProperty, IntProperty
from bpy.types import Operator, Panel, PropertyGroup

# Define camera items as a module-level constant
CAMERA_ITEMS = [
    ("0", "Built-in Camera", "Use built-in camera"),
    ("1", "External Camera", "Use external camera if available")
]

class FaceTrackerProperties(PropertyGroup):
    camera_active: BoolProperty(
        name="Camera Active",
        description="Whether the camera capture is active",
        default=False
    )
    
    selected_camera: EnumProperty(
        name="Camera",
        description="Select camera device",
        items=CAMERA_ITEMS,  # Use static list
        default="0"
    )
    
    show_camera_feed: BoolProperty(
        name="Show Camera Feed",
        description="Display camera feed in viewport",
        default=True
    )
    
    preview_scale: FloatProperty(
        name="Preview Scale",
        description="Scale of the camera preview in viewport",
        default=0.3,
        min=0.1,
        max=1.0
    )
    
    preview_opacity: FloatProperty(
        name="Preview Opacity",
        description="Opacity of the camera preview",
        default=0.8,
        min=0.1,
        max=1.0
    )

class FACETRACKER_OT_GenerateMesh(Operator):
    """Generate a basic face mesh for tracking"""
    bl_idname = "facetracker.generate_mesh"
    bl_label = "Generate Face Mesh"
    bl_options = {'REGISTER', 'UNDO'}
    
    def execute(self, context):
        try:
            # Create a new mesh
            mesh = bpy.data.meshes.new(name="FaceMesh")
            obj = bpy.data.objects.new(name="FaceObject", object_data=mesh)
            bpy.context.collection.objects.link(obj)
            
            # Create a basic face mesh
            bm = bmesh.new()
            
            # Add vertices for a basic face shape (plane for now)
            verts = [
                bm.verts.new((-1, -1, 0)),  # Bottom left
                bm.verts.new((1, -1, 0)),   # Bottom right
                bm.verts.new((1, 1, 0)),    # Top right
                bm.verts.new((-1, 1, 0)),   # Top left
            ]
            
            # Create a face from the vertices
            bm.faces.new(verts)
            
            # Write the mesh data
            bm.to_mesh(mesh)
            bm.free()
            mesh.update()
            
            # Select and make active
            bpy.ops.object.select_all(action='DESELECT')
            obj.select_set(True)
            context.view_layer.objects.active = obj
            
            # Add material with transparency
            mat = bpy.data.materials.new(name="FaceTrackerMaterial")
            mat.use_nodes = True
            mat.blend_method = 'BLEND'
            mat.shadow_method = 'NONE'
            
            # Set up basic transparent material
            nodes = mat.node_tree.nodes
            links = mat.node_tree.links
            
            # Clear default nodes
            nodes.clear()
            
            # Create nodes
            node_bsdf = nodes.new('ShaderNodeBsdfPrincipled')
            node_output = nodes.new('ShaderNodeOutputMaterial')
            
            # Set node locations
            node_bsdf.location = (-300, 0)
            node_output.location = (0, 0)
            
            # Connect nodes
            links.new(node_bsdf.outputs['BSDF'], node_output.inputs['Surface'])
            
            # Set material properties
            node_bsdf.inputs['Base Color'].default_value = (0.8, 0.8, 1.0, 1.0)
            node_bsdf.inputs['Alpha'].default_value = 0.5
            
            # Assign material to object
            if obj.data.materials:
                obj.data.materials[0] = mat
            else:
                obj.data.materials.append(mat)
            
            # Add shape keys for facial expressions
            obj.shape_key_add(name='Basis')
            
            # Add some basic shape keys for mouth movements
            shapes = ['A', 'E', 'I', 'O', 'U']
            for shape in shapes:
                sk = obj.shape_key_add(name=f'Mouth_{shape}')
                sk.value = 0.0
            
            self.report({'INFO'}, "Generated face mesh with shape keys")
            return {'FINISHED'}
        except Exception as e:
            self.report({'ERROR'}, f"Failed to generate mesh: {str(e)}")
            return {'CANCELLED'}

def draw_callback_px(self, context):
    """Draw camera feed in the viewport"""
    try:
        if not self._frame_queue.empty():
            frame = self._frame_queue.get()
            if frame is not None:
                # Convert frame to shader-compatible format
                frame_rgba = cv2.cvtColor(frame, cv2.COLOR_BGR2RGBA)
                frame_rgba = np.flipud(frame_rgba)  # Flip vertically for OpenGL
                
                # Create or update texture
                if self._texture is None:
                    self._texture = gpu.texture.from_image(frame_rgba)
                else:
                    try:
                        self._texture.write(frame_rgba)
                    except Exception as e:
                        print(f"Texture update error: {str(e)}")
                        # Recreate texture if update fails
                        self._texture = gpu.texture.from_image(frame_rgba)
                
                # Set up shader for drawing
                if self._shader is None:
                    vertex_shader = '''
                        uniform mat4 ModelViewProjectionMatrix;
                        in vec2 pos;
                        in vec2 texCoord;
                        out vec2 texCoord_interp;
                        
                        void main() {
                            gl_Position = ModelViewProjectionMatrix * vec4(pos.xy, 0.0, 1.0);
                            texCoord_interp = texCoord;
                        }
                    '''
                    
                    fragment_shader = '''
                        uniform sampler2D image;
                        uniform float opacity;
                        in vec2 texCoord_interp;
                        out vec4 fragColor;
                        
                        void main() {
                            vec4 color = texture(image, texCoord_interp);
                            fragColor = vec4(color.rgb, color.a * opacity);
                        }
                    '''
                    
                    try:
                        self._shader = gpu.types.GPUShader(vertex_shader, fragment_shader)
                    except Exception as e:
                        print(f"Shader creation error: {str(e)}")
                        return
                
                # Calculate preview dimensions
                region = context.region
                scale = context.scene.facetracker.preview_scale
                aspect = frame_rgba.shape[1] / frame_rgba.shape[0]  # width/height
                preview_height = region.height * scale
                preview_width = preview_height * aspect
                
                # Position preview in top-right corner with padding
                padding = 20
                x = region.width - preview_width - padding
                y = region.height - padding
                
                # Create batch if it doesn't exist or if dimensions changed
                vertices = (
                    (x, y - preview_height),  # bottom-left
                    (x + preview_width, y - preview_height),  # bottom-right
                    (x + preview_width, y),  # top-right
                    (x, y)  # top-left
                )
                
                self._batch = batch_for_shader(
                    self._shader, 'TRI_FAN',
                    {
                        "pos": vertices,
                        "texCoord": ((0, 1), (1, 1), (1, 0), (0, 0)),  # Flip UV coords
                    }
                )
                
                # Draw the preview
                if context.scene.facetracker.show_camera_feed:
                    gpu.state.blend_set('ALPHA')
                    self._shader.bind()
                    
                    matrix = gpu.matrix.get_projection_matrix()
                    self._shader.uniform_float("ModelViewProjectionMatrix", matrix)
                    self._shader.uniform_float("opacity", context.scene.facetracker.preview_opacity)
                    self._shader.uniform_sampler("image", self._texture)
                    
                    self._batch.draw(self._shader)
                    gpu.state.blend_set('NONE')
                    
    except Exception as e:
        print(f"Error in draw_callback_px: {str(e)}")
        print(traceback.format_exc())

def check_camera_permissions():
    """Check if we have camera permissions on macOS"""
    if sys.platform != "darwin":
        return True
        
    try:
        import subprocess
        result = subprocess.run([
            'tccutil', 'query', 'Camera', 'com.blender.blender'
        ], capture_output=True, text=True)
        return "allowed" in result.stdout.lower()
    except Exception:
        return False

def request_camera_permissions():
    """Request camera permissions on macOS"""
    if sys.platform != "darwin":
        return True
        
    try:
        import subprocess
        subprocess.run([
            'osascript',
            '-e',
            'tell application "System Events" to tell process "Blender" to every window'
        ], check=True)
        return True
    except Exception:
        return False

class FACETRACKER_OT_StartCamera(Operator):
    """Start camera capture and face tracking"""
    bl_idname = "facetracker.start_camera"
    bl_label = "Start Camera"
    bl_options = {'REGISTER', 'UNDO'}
    
    _timer = None
    _cap = None
    _frame_queue = Queue(maxsize=1)
    _tracking_thread = None
    _stop_thread = False
    _image = None
    _handle = None
    _texture = None
    _batch = None
    _shader = None
    
    def init_camera(self, camera_idx):
        """Initialize camera with proper settings"""
        try:
            if sys.platform == "darwin":
                # On macOS, explicitly use AVFoundation backend
                print("Initializing camera with AVFoundation backend...")
                
                # First try to force a permission prompt with AVFoundation
                for i in range(2):  # Try both 0 and 1 for built-in and external cameras
                    test_cap = cv2.VideoCapture(cv2.CAP_AVFOUNDATION + i)
                    if test_cap.isOpened():
                        print(f"Successfully opened test camera {i}")
                        ret, frame = test_cap.read()  # This should trigger permission prompt
                        test_cap.release()
                        if ret:
                            print("Successfully read test frame")
                            break
                
                # Now try to open the actual camera
                self._cap = cv2.VideoCapture(cv2.CAP_AVFOUNDATION + camera_idx)
                
                if not self._cap.isOpened():
                    print("Failed to open camera with AVFoundation, trying default...")
                    self._cap = cv2.VideoCapture(camera_idx)  # Try default backend
                
                if self._cap.isOpened():
                    print("Camera opened successfully")
                    
                    # Configure camera
                    self._cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'))
                    self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
                    self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
                    self._cap.set(cv2.CAP_PROP_FPS, 30)
                    
                    # Verify we can read frames
                    for _ in range(10):  # Try up to 10 times
                        ret, frame = self._cap.read()
                        if ret and frame is not None and frame.size > 0:
                            print(f"Successfully reading frames: {frame.shape}")
                            return True
                        time.sleep(0.1)
                    
                    print("Failed to read frames from camera")
                    self._cap.release()
                    return False
                
                print("Failed to open camera")
                return False
                
            else:
                # Non-macOS systems
                self._cap = cv2.VideoCapture(camera_idx)
                return self._cap.isOpened()
            
        except Exception as e:
            print(f"Camera initialization error: {str(e)}")
            if self._cap:
                self._cap.release()
            return False
    
    def capture_frames(self):
        print("Starting capture thread...")
        while not self._stop_thread:
            if self._cap is None or not self._cap.isOpened():
                print("Camera not opened, waiting...")
                time.sleep(0.1)
                continue
                
            ret, frame = self._cap.read()
            if ret:
                print("Frame captured successfully")
                # Flip frame horizontally for mirror effect
                frame = cv2.flip(frame, 1)
                
                if self._frame_queue.full():
                    self._frame_queue.get()
                self._frame_queue.put(frame)
            else:
                print("Failed to capture frame")
            time.sleep(0.016)  # ~60fps
    
    def modal(self, context, event):
        if event.type == 'TIMER':
            try:
                if not self._frame_queue.empty():
                    frame = self._frame_queue.get()
                    
                    # Convert frame to grayscale for face detection
                    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                    
                    # Load the face cascade classifier
                    cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
                    face_cascade = cv2.CascadeClassifier(cascade_path)
                    
                    # Detect faces
                    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
                    
                    for (x, y, w, h) in faces:
                        print(f"Face detected at: {x}, {y}, {w}, {h}")
                        # Store face coordinates for mesh updates
                        context.scene.facetracker_face_coords = (x, y, w, h)
                
                # Request redraw for camera feed
                context.area.tag_redraw()
            except Exception as e:
                print(f"Error in modal: {str(e)}")
            return {'PASS_THROUGH'}
        
        if event.type == 'ESC' or not context.scene.facetracker.camera_active:
            self.cancel(context)
            return {'CANCELLED'}
        
        return {'PASS_THROUGH'}
    
    def execute(self, context):
        print("Starting camera...")
        if not OPENCV_AVAILABLE:
            self.report({'ERROR'}, "OpenCV is not available. Please install opencv-python package.")
            return {'CANCELLED'}
        
        # Get selected camera index
        camera_idx = int(context.scene.facetracker.selected_camera)
        print(f"Using camera index: {camera_idx}")
        
        # Try to open the camera
        try:
            print("Initializing camera...")
            if not self.init_camera(camera_idx):
                if sys.platform == "darwin":
                    self.report({'ERROR'},
                        "Failed to access camera. Please try these steps:\n"
                        "1. Quit Blender\n"
                        "2. Open Terminal\n"
                        "3. Run these commands:\n"
                        "   sudo killall VDCAssistant\n"
                        "   sudo killall AppleCameraAssistant\n"
                        "   tccutil reset Camera\n"
                        "4. Open System Settings → Privacy & Security\n"
                        "5. Start Blender and try again\n"
                        "\nIf that doesn't work, try restarting your computer."
                    )
                else:
                    self.report({'ERROR'}, "Failed to open camera. Please check if it's connected and not in use by another application.")
                return {'CANCELLED'}
            
            print("Camera initialized successfully")
            context.scene.facetracker.camera_active = True
            
            # Add the viewport drawing callback
            args = (self, context)
            self._handle = bpy.types.SpaceView3D.draw_handler_add(draw_callback_px, args, 'WINDOW', 'POST_PIXEL')
            print("Added viewport drawing callback")
            
            # Start capture thread
            self._stop_thread = False
            self._tracking_thread = Thread(target=self.capture_frames)
            self._tracking_thread.daemon = True
            self._tracking_thread.start()
            print("Started capture thread")
            
            wm = context.window_manager
            self._timer = wm.event_timer_add(0.016, window=context.window)
            wm.modal_handler_add(self)
            print("Added modal handler")
            
            return {'RUNNING_MODAL'}
            
        except Exception as e:
            print(f"Camera initialization error: {str(e)}")
            print(traceback.format_exc())
            if self._cap:
                self._cap.release()
            self.report({'ERROR'}, f"Camera error: {str(e)}")
            return {'CANCELLED'}

class FACETRACKER_OT_StopCamera(Operator):
    """Stop camera capture"""
    bl_idname = "facetracker.stop_camera"
    bl_label = "Stop Camera"
    
    def execute(self, context):
        context.scene.facetracker.camera_active = False
        return {'FINISHED'}

class FACETRACKER_PT_MainPanel(Panel):
    """Main panel for the Face Tracker add-on"""
    bl_label = "Face Tracker"
    bl_idname = "FACETRACKER_PT_MainPanel"
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = 'Face Tracker'
    
    def draw(self, context):
        layout = self.layout
        scene = context.scene
        
        # Check dependencies
        if not OPENCV_AVAILABLE:
            box = layout.box()
            box.label(text="Missing Dependencies", icon='ERROR')
            box.label(text="Please install: opencv-python")
            return
        
        # Camera controls
        box = layout.box()
        box.label(text="Camera Control")
        
        row = box.row()
        row.prop(scene.facetracker, "selected_camera", text="Camera")
        
        if not scene.facetracker.camera_active:
            box.operator("facetracker.start_camera", text="Start Camera", icon='PLAY')
        else:
            box.operator("facetracker.stop_camera", text="Stop Camera", icon='PAUSE')
            box.prop(scene.facetracker, "show_camera_feed", text="Show Camera Feed")
            
            # Preview settings
            if scene.facetracker.show_camera_feed:
                box.prop(scene.facetracker, "preview_scale", text="Preview Scale")
                box.prop(scene.facetracker, "preview_opacity", text="Preview Opacity")
        
        # Mesh generation
        box = layout.box()
        box.label(text="Mesh Generation")
        box.operator("facetracker.generate_mesh", text="Generate Face Mesh", icon='MESH_DATA')

# Update registration to follow Blender 4.3 best practices
classes = (
    FaceTrackerProperties,
    FACETRACKER_OT_GenerateMesh,
    FACETRACKER_OT_StartCamera,
    FACETRACKER_OT_StopCamera,
    FACETRACKER_PT_MainPanel
)

def register():
    for cls in classes:
        try:
            bpy.utils.register_class(cls)
        except Exception as e:
            print(f"Failed to register {cls.__name__}: {str(e)}")
            return
    
    try:
        bpy.types.Scene.facetracker = bpy.props.PointerProperty(type=FaceTrackerProperties)
        bpy.types.Scene.facetracker_face_coords = bpy.props.IntVectorProperty(
            size=4,
            default=(0, 0, 0, 0)
        )
    except Exception as e:
        print(f"Failed to register properties: {str(e)}")
        # Clean up if property registration fails
        for cls in reversed(classes):
            if hasattr(bpy.types, cls.__name__):
                bpy.utils.unregister_class(cls)

def unregister():
    try:
        del bpy.types.Scene.facetracker_face_coords
        del bpy.types.Scene.facetracker
    except Exception as e:
        print(f"Failed to unregister properties: {str(e)}")
    
    for cls in reversed(classes):
        try:
            bpy.utils.unregister_class(cls)
        except Exception as e:
            print(f"Failed to unregister {cls.__name__}: {str(e)}")

if __name__ == "__main__":
    register() 