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
- `-bd, --border_dist`: Distance from ablation border to increase density (default: 10)
- `-bf, --border_density`: Factor to increase keypoint density near borders (default: 2.0)
- (All other original CorrField parameters remain available)

### Input Requirements
- Fixed image (post-ablation CT), ablation mask, and lung mask must have the same dimensions
- Ablation mask should be binary (0 for background, 1 for ablation zone)
- Moving image (pre-ablation CT) can have different dimensions

### Outputs
- Warped pre-ablation CT scan
- CSV file with correspondence points
- Registration transform

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
# Basic usage with default border parameters
python corrfield.py -F post_ablation.nii.gz \
                   -M pre_ablation.nii.gz \
                   -m lung_mask.nii.gz \
                   -A ablation_mask.nii.gz \
                   -O registered_output

# Usage with custom border parameters
python corrfield.py -F post_ablation.nii.gz \
                   -M pre_ablation.nii.gz \
                   -m lung_mask.nii.gz \
                   -A ablation_mask.nii.gz \
                   -O registered_output \
                   -bd 15 \
                   -bf 2.5
```

## Implementation Notes
- The algorithm assumes the ablation mask is in the fixed image space
- Border region is computed using morphological operations
- Keypoint selection is modified only in and around the ablation zone
- Original registration behavior is maintained in regions away from the ablation zone

