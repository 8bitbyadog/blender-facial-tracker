# FaceBuilder Alternative for Blender

A free and open-source alternative to KeenTools FaceBuilder for Blender, allowing you to generate 3D face models from images using OpenCV and DLib.

## Features

- Face detection using OpenCV
- Basic mesh generation from detected faces
- Integration with Blender's UI
- Support for multiple image angles (coming soon)

## Installation

1. Install the required Python packages:
```bash
pip install -r requirements.txt
```

2. In Blender:
   - Go to Edit > Preferences > Add-ons
   - Click "Install" and select the `facebuilder_alternative.py` file
   - Enable the add-on by checking the checkbox

## Usage

1. Open Blender and switch to the 3D Viewport
2. Open the sidebar (press N if not visible)
3. Look for the "FaceBuilder" tab
4. Click "Load Image" to select a face image
5. Use "Generate Face Mesh" to create a basic face mesh

## Development Roadmap

- [x] Basic add-on structure
- [x] Face detection
- [ ] Advanced mesh generation
- [ ] Multi-angle support
- [ ] Texture mapping
- [ ] Auto-rigging

## Requirements

- Blender 3.0 or higher
- Python 3.7 or higher
- OpenCV
- NumPy
- DLib
- SciPy

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details. 