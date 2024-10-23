# Modified CorrField Algorithm for Pre/Post-Ablation CT Registration

## Overview
This repository contains a modified version of the CorrField registration algorithm, specifically adapted for registering pre- and post-ablation CT scans. The algorithm has been enhanced to handle the challenging scenario where the ablation zone in the post-ablation scan has significantly different appearance compared to the corresponding region in the pre-ablation scan.

## Modifications to Original Algorithm

### Key Concept Changes
- Added consideration of ablation zones in post-ablation CT (fixed image)
- Modified keypoint selection strategy around ablation zones
- Introduced new parameters for controlling keypoint density

### Detailed Changes

#### 1. Förstner Keypoint Detection
Added new function `foerstner_kpts_with_exclusion` that:
- Takes additional `ablation_mask` parameter
- Completely excludes keypoints inside ablation zone
- Creates a border region around ablation zone using morphological operations
- Increases keypoint distinctiveness in border region by a configurable factor
- Allows control of border region width

Key implementation features:
```python
# Create border region using dilation
border_region = torch.zeros_like(ablation_mask)
for i in range(1, border_dist + 1):
    dilated = F.max_pool3d(ablation_mask.float(), 
                          kernel_size=2*i+1, 
                          stride=1, 
                          padding=i)
    border_region = torch.logical_or(border_region, 
                                   torch.logical_and(dilated > 0, 
                                                   ablation_mask == 0))

# Modify distinctiveness scores
distinctiveness = torch.where(ablation_mask > 0, 
                            torch.zeros_like(distinctiveness),  # Zero in ablation
                            distinctiveness)  # Original elsewhere

distinctiveness = torch.where(border_region > 0,
                            distinctiveness * border_density,  # Increase near border
                            distinctiveness)  # Original elsewhere
```

#### 2. Main Registration Function
Modified to handle ablation zones through:
- New `corrfield_with_ablation` function
- Integration with modified keypoint detection
- Maintained original registration pipeline with modified keypoint selection
- Returns warped moving image and correspondence points

## Usage

### Command Line Arguments
New required and optional arguments added to the original interface:

```bash
python corrfield.py -F fixed.nii.gz \
                   -M moving.nii.gz \
                   -m mask.nii.gz \
                   -A ablation_mask.nii.gz \
                   -O output \
                   [--save_deformation] \
                   [-bd BORDER_DIST] \
                   [-bf BORDER_DENSITY]
```

#### Required Arguments:
- `-F, --fixed`: Post-ablation CT scan (fixed image)
- `-M, --moving`: Pre-ablation CT scan (moving image)
- `-m, --mask`: Mask for valid regions in fixed image
- `-A, --ablation_mask`: Binary mask of ablation zone in fixed image space
- `-O, --output`: Output name prefix (no filename extension)

#### Optional Arguments:
- `--save_deformation`: Generate and save deformation field files (default: False)
- `-bd, --border_dist`: Distance from ablation border to increase density (default: 10)
- `-bf, --border_density`: Factor to increase keypoint density near borders (default: 2.0)
- (All other original CorrField parameters remain available)

### Input Requirements
- Fixed image (post-ablation CT), ablation mask, and lung mask must have the same dimensions
- Ablation mask should be binary (0 for background, 1 for ablation zone)
- Moving image (pre-ablation CT) can have different dimensions

### Outputs
- `output.nii.gz`: Warped pre-ablation CT scan
- `output.csv`: Correspondence points
- `output_keypoints_fixed.npy`: Fixed keypoints for visualization
- `output_keypoints_moving.npy`: Moving keypoints for visualization
- When `--save_deformation` is used:
  - `output_deformation_magnitude.nii.gz`: Magnitude of deformation field
  - `output_deformation_x.nii.gz`: X component of deformation field
  - `output_deformation_y.nii.gz`: Y component of deformation field
  - `output_deformation_z.nii.gz`: Z component of deformation field

## Keypoint Visualization

A separate visualization tool (`keypoint_viewer.py`) is provided to inspect the registration results and keypoint placement:

```bash
python keypoint_viewer.py \
    --fixed_image fixed.nii.gz \
    --moving_image moving.nii.gz \
    --ablation_mask ablation_mask.nii.gz \
    --fixed_keypoints output_keypoints_fixed.npy \
    --moving_keypoints output_keypoints_moving.npy \
    --axis z \
    --slice_thickness 1
```

### Visualization Features:
- Side-by-side view of fixed and moving images
- Interactive slice navigation with slider and keyboard controls
- Keypoint overlay on respective images
  - Fixed keypoints shown as green dots
  - Moving keypoints shown as red plus signs
- Ablation zone overlay in semi-transparent yellow
- Images rotated 270 degrees clockwise for standard radiological view
- Adjustable slice thickness for keypoint visibility

### Keypoint Viewer Arguments:
- `--fixed_image`: Path to fixed (post-ablation) scan
- `--moving_image`: Path to moving (pre-ablation) scan
- `--ablation_mask`: Path to ablation mask
- `--fixed_keypoints`: Path to fixed keypoints (.npy file)
- `--moving_keypoints`: Path to moving keypoints (.npy file)
- `--axis`: Viewing axis (x, y, or z; default: z)
- `--slice_thickness`: Show points within ±N slices (default: 1)

### Interactive Controls:
- Slider: Manual slice selection
- Arrow keys: Navigate slices (up/right = next, down/left = previous)
- Mouse click: Select which image to control with keyboard
- Each image has its own slice slider for independent navigation

## Parameter Guidelines

### Border Distance (`border_dist`)
- Controls the width of the region around the ablation zone where keypoint density is increased
- Recommended range: 5-15 voxels
- Larger values for bigger ablation zones
- Consider image resolution when setting

### Border Density Factor (`border_density`)
- Controls how much to increase keypoint density near ablation zone
- Recommended range: 1.5-3.0
- Higher values for more challenging cases
- Balance between accuracy and over-fitting

## Performance Considerations
- Minimal additional computation time compared to original algorithm
- Small memory overhead for border region calculation
- No significant impact on overall performance

## Example Usage
```bash
# Basic registration with keypoint saving
python corrfield.py -F post_ablation.nii.gz \
                   -M pre_ablation.nii.gz \
                   -m lung_mask.nii.gz \
                   -A ablation_mask.nii.gz \
                   -O registered_output

# Registration with deformation field saving and custom border parameters
python corrfield.py -F post_ablation.nii.gz \
                   -M pre_ablation.nii.gz \
                   -m lung_mask.nii.gz \
                   -A ablation_mask.nii.gz \
                   -O registered_output \
                   --save_deformation \
                   -bd 15 \
                   -bf 2.5

# Keypoint visualization
python keypoint_viewer.py \
    --fixed_image post_ablation.nii.gz \
    --moving_image pre_ablation.nii.gz \
    --ablation_mask ablation_mask.nii.gz \
    --fixed_keypoints registered_output_keypoints_fixed.npy \
    --moving_keypoints registered_output_keypoints_moving.npy \
    --axis z \
    --slice_thickness 1
```

## Implementation Notes
- The algorithm assumes the ablation mask is in the fixed image space
- Border region is computed using morphological operations
- Keypoint selection is modified only in and around the ablation zone
- Original registration behavior is maintained in regions away from the ablation zone
