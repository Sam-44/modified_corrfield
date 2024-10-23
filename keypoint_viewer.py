import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
import argparse
from scipy.ndimage import rotate

class KeypointViewer:
    def __init__(self, fixed_image_path, moving_image_path, ablation_mask_path=None,
                 fixed_keypoints_path=None, moving_keypoints_path=None):
        """Initialize as before"""
        # Load images
        self.fixed_image = nib.load(fixed_image_path)
        self.moving_image = nib.load(moving_image_path)
        self.fixed_img_data = self.fixed_image.get_fdata()
        self.moving_img_data = self.moving_image.get_fdata()
        
        # Load ablation mask if provided
        if ablation_mask_path:
            self.ablation_mask = nib.load(ablation_mask_path)
            self.ablation_data = self.ablation_mask.get_fdata().astype(bool)  # Ensure binary
        else:
            self.ablation_data = None
        
        # Load keypoints if provided
        self.fixed_keypoints = np.load(fixed_keypoints_path) if fixed_keypoints_path else None
        self.moving_keypoints = np.load(moving_keypoints_path) if moving_keypoints_path else None

    def rotate_points(self, points, angle_degrees, center):
        """Rotate points around center"""
        # Convert to radians
        angle_rad = np.deg2rad(angle_degrees)
        
        # Translate points to origin
        points = points - np.array(center)
        
        # Rotation matrix
        rot_matrix = np.array([
            [np.cos(angle_rad), -np.sin(angle_rad)],
            [np.sin(angle_rad), np.cos(angle_rad)]
        ])
        
        # Rotate points
        points = np.dot(points, rot_matrix.T)
        
        # Translate back
        points = points + np.array(center)
        
        return points
        
    def interactive_plot(self, axis='z', slice_thickness=1):
        """
        Create interactive plot with slice scrolling for both volumes.
        
        Args:
            axis: Axis along which to take slices ('x', 'y', or 'z')
            slice_thickness: Points within this range of the slice will be shown
        """
        self.axis = axis
        self.slice_thickness = slice_thickness

        # Create figure with two subplots side by side
        self.fig, (self.ax1, self.ax2) = plt.subplots(1, 2, figsize=(20, 10))
        
        # Determine maximum slice number for each volume
        max_slice_fixed = self.fixed_img_data.shape[{'x':0, 'y':1, 'z':2}[axis]] - 1
        max_slice_moving = self.moving_img_data.shape[{'x':0, 'y':1, 'z':2}[axis]] - 1
        
        initial_slice_fixed = max_slice_fixed // 2
        initial_slice_moving = max_slice_moving // 2

        # Create slider axes
        self.fig.subplots_adjust(bottom=0.15)  # Make room for sliders
        slider_ax_fixed = plt.axes([0.1, 0.05, 0.35, 0.03])
        slider_ax_moving = plt.axes([0.55, 0.05, 0.35, 0.03])
        
        self.slider_fixed = Slider(slider_ax_fixed, 'Fixed Slice', 0, max_slice_fixed, 
                                 valinit=initial_slice_fixed, valstep=1)
        self.slider_moving = Slider(slider_ax_moving, 'Moving Slice', 0, max_slice_moving, 
                                  valinit=initial_slice_moving, valstep=1)
        
        # Plot initial slices
        self.update_fixed_plot(initial_slice_fixed)
        self.update_moving_plot(initial_slice_moving)
        
        # Connect sliders to update functions
        self.slider_fixed.on_changed(self.update_fixed_plot)
        self.slider_moving.on_changed(self.update_moving_plot)
        
        # Connect keyboard events
        self.selected_plot = 'fixed'  # Track which plot is active for keyboard navigation
        self.fig.canvas.mpl_connect('key_press_event', self.on_key)
        self.fig.canvas.mpl_connect('button_press_event', self.on_click)
        
        plt.show()

    def update_fixed_plot(self, slice_num):
        """Update fixed image plot"""
        self.ax1.clear()
        
        # Get slice data based on axis
        if self.axis == 'z':
            img_slice = self.fixed_img_data[:, :, int(slice_num)]
            ablation_slice = self.ablation_data[:, :, int(slice_num)] if self.ablation_data is not None else None
            coord_idx = 2
            plot_idx = [0, 1]
        elif self.axis == 'y':
            img_slice = self.fixed_img_data[:, int(slice_num), :]
            ablation_slice = self.ablation_data[:, int(slice_num), :] if self.ablation_data is not None else None
            coord_idx = 1
            plot_idx = [0, 2]
        else:  # x
            img_slice = self.fixed_img_data[int(slice_num), :, :]
            ablation_slice = self.ablation_data[int(slice_num), :, :] if self.ablation_data is not None else None
            coord_idx = 0
            plot_idx = [1, 2]

        # Rotate image and mask 270 degrees clockwise
        img_slice = rotate(img_slice, 270, reshape=False)
        if ablation_slice is not None:
            ablation_slice = rotate(ablation_slice, 270, reshape=False)

        # Plot rotated image slice
        self.ax1.imshow(img_slice, cmap='gray')
        
        # Overlay ablation mask if available
        if ablation_slice is not None:
            # Create a colored overlay for the ablation mask
            overlay = np.zeros((*ablation_slice.shape, 4))  # RGBA
            overlay[ablation_slice > 0] = [1, 1, 0, 0.3]  # Yellow with 0.3 alpha
            self.ax1.imshow(overlay)

        # Plot fixed keypoints with rotation
        if self.fixed_keypoints is not None:
            fixed_pts = self.fixed_keypoints[0]
            mask = np.abs(fixed_pts[:, coord_idx] - slice_num) <= self.slice_thickness
            if np.any(mask):
                points = fixed_pts[mask][:, plot_idx]
                # Rotate points around image center
                center = (img_slice.shape[0]/2, img_slice.shape[1]/2)
                rotated_points = self.rotate_points(points, 270, center)
                
                self.ax1.scatter(rotated_points[:, 1], 
                               rotated_points[:, 0], 
                               c='lime', marker='.', s=25,
                               label=f'Fixed keypoints (±{self.slice_thickness} slice)')
                self.ax1.legend()

        self.ax1.set_title(f'Fixed Volume - Slice {int(slice_num)} ({self.axis}-axis)')
        self.ax1.axis('image')
        self.fig.canvas.draw_idle()

    def update_moving_plot(self, slice_num):
        """Update moving image plot"""
        self.ax2.clear()
        
        # Get slice data based on axis
        if self.axis == 'z':
            img_slice = self.moving_img_data[:, :, int(slice_num)]
            coord_idx = 2
            plot_idx = [0, 1]
        elif self.axis == 'y':
            img_slice = self.moving_img_data[:, int(slice_num), :]
            coord_idx = 1
            plot_idx = [0, 2]
        else:  # x
            img_slice = self.moving_img_data[int(slice_num), :, :]
            coord_idx = 0
            plot_idx = [1, 2]

        # Rotate image 270 degrees clockwise
        img_slice = rotate(img_slice, 270, reshape=False)

        # Plot rotated image slice
        self.ax2.imshow(img_slice, cmap='gray')

        # Plot moving keypoints with rotation
        if self.moving_keypoints is not None:
            moving_pts = self.moving_keypoints[0]
            mask = np.abs(moving_pts[:, coord_idx] - slice_num) <= self.slice_thickness
            if np.any(mask):
                points = moving_pts[mask][:, plot_idx]
                # Rotate points around image center
                center = (img_slice.shape[0]/2, img_slice.shape[1]/2)
                rotated_points = self.rotate_points(points, 270, center)
                
                self.ax2.scatter(rotated_points[:, 1], 
                               rotated_points[:, 0], 
                               c='red', marker='+', s=25,
                               linewidths=0.5,
                               label=f'Moving keypoints (±{self.slice_thickness} slice)')
                self.ax2.legend()

        self.ax2.set_title(f'Moving Volume - Slice {int(slice_num)} ({self.axis}-axis)')
        self.ax2.axis('image')
        self.fig.canvas.draw_idle()

    def on_click(self, event):
        """Handle mouse clicks to determine active plot"""
        if event.inaxes == self.ax1:
            self.selected_plot = 'fixed'
        elif event.inaxes == self.ax2:
            self.selected_plot = 'moving'

    def on_key(self, event):
        """Handle keyboard events for slice navigation"""
        if self.selected_plot == 'fixed':
            slider = self.slider_fixed
        else:
            slider = self.slider_moving
            
        if event.key == 'up' or event.key == 'right':
            slider.set_val(min(slider.val + 1, slider.valmax))
        elif event.key == 'down' or event.key == 'left':
            slider.set_val(max(slider.val - 1, slider.valmin))

def main():
    parser = argparse.ArgumentParser(description='View keypoints overlaid on scans')
    parser.add_argument('--fixed_image', required=True,
                      help='Path to fixed scan (.nii/.nii.gz)')
    parser.add_argument('--moving_image', required=True,
                      help='Path to moving scan (.nii/.nii.gz)')
    parser.add_argument('--ablation_mask',
                      help='Path to ablation mask (.nii/.nii.gz)')
    parser.add_argument('--fixed_keypoints',
                      help='Path to fixed keypoints (.npy)')
    parser.add_argument('--moving_keypoints',
                      help='Path to moving keypoints (.npy)')
    parser.add_argument('--axis', default='z', choices=['x', 'y', 'z'],
                      help='Viewing axis (default: z)')
    parser.add_argument('--slice_thickness', type=int, default=1,
                      help='Show points within ±N slices (default: 1)')
    
    args = parser.parse_args()
    
    viewer = KeypointViewer(
        args.fixed_image,
        args.moving_image,
        args.ablation_mask,
        args.fixed_keypoints,
        args.moving_keypoints
    )
    
    viewer.interactive_plot(
        axis=args.axis,
        slice_thickness=args.slice_thickness
    )

if __name__ == '__main__':
    main()