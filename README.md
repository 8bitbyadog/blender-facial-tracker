# Blender Face Tracker

A real-time facial motion capture add-on for Blender 4.3+. This add-on provides tools for capturing facial movements using your computer's camera and applying them to 3D models in real-time.

## Features

- Real-time camera feed in the Blender viewport
- Face detection using OpenCV
- Customizable preview settings (scale and opacity)
- Support for multiple cameras
- Basic face mesh generation with shape keys

## Requirements

- Blender 4.3.2 or newer
- Python 3.10+
- OpenCV Python package (`opencv-python`)

## Installation

1. Download the latest release or clone this repository
2. Copy `src/face_tracker.py` to your Blender addons directory:
   - Windows: `%APPDATA%\Blender Foundation\Blender\4.3\scripts\addons\`
   - macOS: `~/Library/Application Support/Blender/4.3/scripts/addons/`
   - Linux: `~/.config/blender/4.3/scripts/addons/`
3. Open Blender and go to Edit → Preferences → Add-ons
4. Search for "Face Tracker" and enable the add-on

## Usage

1. In the 3D Viewport, open the sidebar (N key) and look for the "Face Tracker" tab
2. Select your camera from the dropdown menu
3. Click "Start Camera" to begin capturing
4. Use the preview scale and opacity sliders to adjust the camera feed display
5. Click "Generate Face Mesh" to create a basic face mesh with shape keys

## Troubleshooting

### Camera Access on macOS
If you encounter camera access issues on macOS:
1. Quit Blender
2. Open Terminal and run:
   ```bash
   sudo killall VDCAssistant
   sudo killall AppleCameraAssistant
   tccutil reset Camera
   ```
3. Open System Settings → Privacy & Security → Camera
4. Start Blender and try again

## License

[MIT License](LICENSE)

## Author

8bitbyadog 