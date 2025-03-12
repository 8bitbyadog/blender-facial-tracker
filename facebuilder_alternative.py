bl_info = {
    "name": "FaceBuilder Alternative",
    "author": "Your Name",
    "version": (1, 0, 0),
    "blender": (3, 0, 0),
    "location": "View3D > Sidebar > FaceBuilder",
    "description": "Generate 3D face models from images using OpenCV and DLib",
    "category": "Object",
}

import bpy
import bmesh
import os
import sys
from bpy.props import StringProperty, BoolProperty, FloatProperty
from bpy.types import Operator, Panel

def check_dependencies():
    """Check if all required packages are installed"""
    missing_packages = []
    
    try:
        import cv2
    except ImportError:
        missing_packages.append("opencv-python")
    
    try:
        import numpy
    except ImportError:
        missing_packages.append("numpy")
    
    try:
        import dlib
    except ImportError:
        missing_packages.append("dlib")
    
    try:
        import scipy
    except ImportError:
        missing_packages.append("scipy")
    
    return missing_packages

class FACEBUILDER_OT_LoadImage(Operator):
    """Load and process an image for face detection"""
    bl_idname = "facebuilder.load_image"
    bl_label = "Load Image"
    bl_options = {'REGISTER', 'UNDO'}
    
    filepath: StringProperty(
        subtype='FILE_PATH',
        description="Path to the image file"
    )
    
    def execute(self, context):
        # Check dependencies first
        missing_packages = check_dependencies()
        if missing_packages:
            self.report({'ERROR'}, f"Missing required packages: {', '.join(missing_packages)}")
            return {'CANCELLED'}
            
        try:
            import cv2
            import numpy as np
        except ImportError as e:
            self.report({'ERROR'}, f"Failed to import required packages: {str(e)}")
            return {'CANCELLED'}
            
        if not os.path.exists(self.filepath):
            self.report({'ERROR'}, "File not found")
            return {'CANCELLED'}
            
        # Load and process image
        image = cv2.imread(self.filepath)
        if image is None:
            self.report({'ERROR'}, "Failed to load image")
            return {'CANCELLED'}
            
        # Convert to grayscale for face detection
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Load the face cascade classifier
        cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        if not os.path.exists(cascade_path):
            self.report({'ERROR'}, f"Cascade file not found at: {cascade_path}")
            return {'CANCELLED'}
            
        face_cascade = cv2.CascadeClassifier(cascade_path)
        if face_cascade.empty():
            self.report({'ERROR'}, "Failed to load face cascade classifier")
            return {'CANCELLED'}
        
        # Detect faces
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)
        
        if len(faces) == 0:
            self.report({'WARNING'}, "No faces detected in image")
            return {'CANCELLED'}
            
        # Store the first detected face coordinates
        context.scene.facebuilder_face_coords = faces[0]
        
        self.report({'INFO'}, f"Detected {len(faces)} faces")
        return {'FINISHED'}
    
    def invoke(self, context, event):
        context.window_manager.fileselect_add(self)
        return {'RUNNING_MODAL'}

class FACEBUILDER_OT_GenerateMesh(Operator):
    """Generate a basic face mesh from detected landmarks"""
    bl_idname = "facebuilder.generate_mesh"
    bl_label = "Generate Face Mesh"
    bl_options = {'REGISTER', 'UNDO'}
    
    def execute(self, context):
        try:
            # Create a new mesh
            mesh = bpy.data.meshes.new(name="FaceMesh")
            obj = bpy.data.objects.new(name="FaceObject", object_data=mesh)
            bpy.context.collection.objects.link(obj)
            
            # Create a basic face mesh (placeholder)
            bm = bmesh.new()
            
            # Add vertices for a basic face shape
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
            mesh.update()
            
            self.report({'INFO'}, "Generated basic face mesh")
            return {'FINISHED'}
        except Exception as e:
            self.report({'ERROR'}, f"Failed to generate mesh: {str(e)}")
            return {'CANCELLED'}

class FACEBUILDER_PT_MainPanel(Panel):
    """Main panel for the FaceBuilder add-on"""
    bl_label = "FaceBuilder Alternative"
    bl_idname = "FACEBUILDER_PT_MainPanel"
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = 'FaceBuilder'
    
    def draw(self, context):
        layout = self.layout
        
        # Check dependencies
        missing_packages = check_dependencies()
        if missing_packages:
            box = layout.box()
            box.label(text="Missing Dependencies", icon='ERROR')
            box.label(text=f"Please install: {', '.join(missing_packages)}")
            return
        
        # Image loading section
        box = layout.box()
        box.label(text="Image Processing")
        box.operator("facebuilder.load_image", text="Load Image", icon='IMAGE_DATA')
        
        # Mesh generation section
        box = layout.box()
        box.label(text="Mesh Generation")
        box.operator("facebuilder.generate_mesh", text="Generate Face Mesh", icon='MESH_DATA')

def register():
    bpy.types.Scene.facebuilder_face_coords = bpy.props.IntVectorProperty(
        name="Face Coordinates",
        description="Coordinates of detected face",
        size=4,
        default=(0, 0, 0, 0)
    )
    for cls in [FACEBUILDER_OT_LoadImage, FACEBUILDER_OT_GenerateMesh, FACEBUILDER_PT_MainPanel]:
        bpy.utils.register_class(cls)

def unregister():
    del bpy.types.Scene.facebuilder_face_coords
    for cls in [FACEBUILDER_OT_LoadImage, FACEBUILDER_OT_GenerateMesh, FACEBUILDER_PT_MainPanel]:
        bpy.utils.unregister_class(cls)

if __name__ == "__main__":
    register() 