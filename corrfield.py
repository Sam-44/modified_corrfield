#!/usr/bin/env python

import argparse
import nibabel as nib
import numpy as np
import os
import sys
sys.path.append('..')
import time
import torch
import torch.nn.functional as F

from thin_plate_spline import *
from foerstner import *
from utils import *
from mindssc import *
from similarity import *
from belief_propagation import *
from graphs import *

#def compute_marginals(kpts_fix, img_fix, mind_fix, mind_mov, alpha, beta, disp_radius, disp_step, patch_radius):
#    cost = alpha * ssd(kpts_fix, mind_fix, mind_mov, disp_radius, disp_step, patch_radius)
#        
#    dist = kpts_dist(kpts_fix, img_fix, beta)
#    edges, level = minimum_spanning_tree(dist)
#    marginals = tbp(cost, edges, level, dist)
#    
#    return marginals

def compute_marginals(kpts_fix, img_fix, mind_fix, mind_mov, alpha, beta, disp_radius, disp_step, patch_radius):
    cost = alpha * ssd(kpts_fix, mind_fix, mind_mov, disp_radius, disp_step, patch_radius)

    k = 32
    while True:
        k *= 2
        try:
            dist = kpts_dist(kpts_fix, img_fix, beta, k)
            edges, level = minimum_spanning_tree(dist)
            break
        except:
            pass
    marginals = tbp(cost, edges, level, dist)

    return marginals

def corrfield_with_ablation(img_fix, mask_fix, img_mov, ablation_mask, alpha, beta, gamma, delta, lambd, sigma, sigma1, L, N, Q, R, T, border_dist=10, border_density=2):
    """
    Modified CorrField registration that accounts for ablation zone.
    
    Parameters:
    - img_fix: Fixed (post-ablation) image
    - mask_fix: Valid regions mask for fixed image
    - img_mov: Moving (pre-ablation) image
    - ablation_mask: Binary mask of ablation zone in fixed image space
    - border_dist: Distance from ablation border to increase density (in voxels)
    - border_density: Factor to increase keypoint density near borders
    """
    device = img_fix.device
    _, _, D, H, W = img_fix.shape
    
    print('Compute fixed MIND features ...', end =" ")
    torch.cuda.synchronize()
    t0 = time.time()
    mind_fix = mindssc(img_fix, delta, sigma1)
    torch.cuda.synchronize()
    t1 = time.time()
    print('finished ({:.2f} s).'.format(t1-t0))
        
    dense_flow = torch.zeros((1, D, H, W, 3), device=device)
    img_mov_warped = img_mov
    
    for i in range(len(L)):
        print('Stage {}/{}'.format(i + 1, len(L)))
        print('    search radius: {}'.format(L[i]))
        print('      cube length: {}'.format(N[i]))
        print('     quantisation: {}'.format(Q[i]))
        print('     patch radius: {}'.format(R[i]))
        print('        transform: {}'.format(T[i]))
        
        disp = get_disp(Q[i], L[i], (D, H, W), device=device)
        
        print('    Compute moving MIND features ...', end =" ")
        torch.cuda.synchronize()
        t0 = time.time()
        mind_mov = mindssc(img_mov_warped, delta, sigma1)
        torch.cuda.synchronize()
        t1 = time.time()
        print('finished ({:.2f} s).'.format(t1-t0))
        
        torch.cuda.synchronize()
        t0 = time.time()
        # Use modified keypoint detection with fixed ablation mask
        kpts_fix = foerstner_kpts_with_exclusion(
            img_fix, 
            mask_fix, 
            ablation_mask,  # Fixed ablation mask
            sigma, 
            N[i],
            border_dist=border_dist,
            border_density=border_density
        )
        torch.cuda.synchronize()
        t1 = time.time()
        print('    {} fixed keypoints extracted ({:.2f} s).'.format(kpts_fix.shape[1], t1-t0))

        print('    Compute forward marginals ...', end =" ")
        torch.cuda.synchronize()
        t0 = time.time()
        marginalsf = compute_marginals(kpts_fix, img_fix, mind_fix, mind_mov, alpha, beta, L[i], Q[i], R[i])
        torch.cuda.synchronize()
        t1 = time.time()
        print('finished ({:.2f} s).'.format(t1-t0))

        flow = (F.softmax(-gamma * marginalsf.view(1, kpts_fix.shape[1], -1, 1), dim=2) * 
               disp.view(1, 1, -1, 3)).sum(2)
        
        kpts_mov = kpts_fix + flow

        print('    Compute symmetric backward marginals ...', end =" ")
        torch.cuda.synchronize()
        t0 = time.time()
        marginalsb = compute_marginals(kpts_mov, img_fix, mind_mov, mind_fix, alpha, beta, L[i], Q[i], R[i])
        torch.cuda.synchronize()
        t1 = time.time()
        print('finished ({:.2f} s).'.format(t1-t0))

        marginals = 0.5 * (marginalsf.view(1, kpts_fix.shape[1], -1) + 
                          marginalsb.view(1, kpts_fix.shape[1], -1).flip(2))
        
        flow = (F.softmax(-gamma * marginals.view(1, kpts_fix.shape[1], -1, 1), dim=2) * 
               disp.view(1, 1, -1, 3)).sum(2)
        
        torch.cuda.synchronize()
        t0 = time.time()
        if T[i] == 'r':
            print('    Find rigid transform ...', end =" ")
            rigid = compute_rigid_transform(kpts_fix, kpts_fix + flow)
            dense_flow_ = F.affine_grid(rigid[:, :3, :] - torch.eye(3, 4, device=device).unsqueeze(0), 
                                      (1, 1, D, H, W), align_corners=True)
        elif T[i] == 'n':
            print('    Dense thin plate spline interpolation ...', end =" ")
            dense_flow_ = thin_plate_dense(kpts_fix, flow, (D, H, W), 3, lambd)
        torch.cuda.synchronize()
        t1 = time.time()
        print('finished ({:.2f} s).'.format(t1-t0))
        
        dense_flow += dense_flow_
        
        # Update moving image
        img_mov_warped = F.grid_sample(
            img_mov, 
            F.affine_grid(torch.eye(3, 4, dtype=img_mov.dtype, device=device).unsqueeze(0), 
                         (1, 1, D, H, W), align_corners=True) + dense_flow.to(img_mov.dtype), 
            align_corners=True
        )
        
        print()
        
    flow = F.grid_sample(dense_flow.permute(0, 4, 1, 2, 3), 
                        kpts_fix.view(1, 1, 1, -1, 3), 
                        align_corners=True).view(1, 3, -1).permute(0, 2, 1)
    
    # Return dense_flow in addition to other outputs
    return (img_mov_warped, 
            kpts_world(kpts_fix, (D, H, W), align_corners=True), 
            kpts_world(kpts_fix + flow, (D, H, W), align_corners=True),
            dense_flow)  # Add this to return values

def main(args):
    print('Run corrField registration ...')
    print()
    
    device = 'cuda:0'
    
    img_fix_path = args.fixed
    img_mov_path = args.moving
    mask_fix_path = args.mask
    ablation_mask_path = args.ablation_mask
    output_path = args.output

    alpha = float(args.alpha)
    beta = float(args.beta)
    gamma = float(args.gamma)
    delta = int(args.delta)
    lambd = float(args.lambd)
    sigma = float(args.sigma)
    sigma1 = float(args.sigma1)
    border_dist = int(args.border_dist)
    border_density = float(args.border_density)
    
    print(' Fixed image: {}'.format(img_fix_path))
    print('Moving image: {}'.format(img_mov_path))
    print('  Fixed mask: {}'.format(mask_fix_path))
    print('Ablation mask: {}'.format(ablation_mask_path))
    print('Output files: {}.csv/.nii.gz'.format(output_path))
    print('       alpha: {}'.format(alpha))
    print('        beta: {}'.format(beta))
    print('       gamma: {}'.format(gamma))
    print('      lambda: {}'.format(lambd))
    print('       delta: {}'.format(delta))
    print('       sigma: {}'.format(sigma))
    print('      sigma1: {}'.format(sigma1))
    print(' border_dist: {}'.format(border_dist))
    print('border_density: {}'.format(border_density))
    print()
    
    L = [int(l) for l in args.search_radius.split('x')]
    N = [int(n) for n in args.length.split('x')]
    Q = [int(q) for q in args.quantisation.split('x')]
    R = [int(r) for r in args.patch_radius.split('x')]
    T = [t for t in args.transform.split('x')]
    
    # Initialize GPU
    mindssc(torch.zeros(nib.load(img_fix_path).shape).unsqueeze(0).unsqueeze(0).to(device), delta, sigma1)
    
    # Load all images
    img_fix = torch.from_numpy(nib.load(img_fix_path).get_fdata().astype(np.float32)).unsqueeze(0).unsqueeze(0).to(device)
    img_mov = torch.from_numpy(nib.load(img_mov_path).get_fdata().astype(np.float32)).unsqueeze(0).unsqueeze(0).to(device)
    mask_fix = torch.from_numpy(nib.load(mask_fix_path).get_fdata().astype(np.float32)).unsqueeze(0).unsqueeze(0).to(device)
    ablation_mask = torch.from_numpy(nib.load(ablation_mask_path).get_fdata().astype(np.float32)).unsqueeze(0).unsqueeze(0).to(device)
    
    torch.cuda.synchronize()
    t0 = time.time()
    
    img_mov_warped, kpts_fix, kpts_mov_corr, dense_flow = corrfield_with_ablation(
        img_fix, mask_fix, img_mov, ablation_mask,
        alpha, beta, gamma, delta, lambd, sigma, sigma1,
        L, N, Q, R, T,
        border_dist=border_dist,
        border_density=border_density
    )

    torch.cuda.synchronize()
    t1 = time.time()
    
    # Save original outputs
    np.savetxt('{}.csv'.format(output_path), 
               torch.cat([kpts_fix[0], kpts_mov_corr[0]], dim=1).cpu().numpy(), 
               delimiter=",", fmt='%.3f')
    
    np.save('{}_keypoints_fixed.npy'.format(output_path), 
            kpts_fix.cpu().numpy())
    np.save('{}_keypoints_moving.npy'.format(output_path), 
            kpts_mov_corr.cpu().numpy())
    
    nib.save(nib.Nifti1Image(np.squeeze(img_mov_warped.cpu().numpy()), 
                            nib.load(img_fix_path).affine), 
             '{}.nii.gz'.format(output_path))
    
    print('Files written:')
    print('  {}.csv'.format(output_path))
    print('  {}.nii.gz'.format(output_path))
    
    if args.save_deformation:
    
        # Compute and save deformation field components
        magnitude, dx, dy, dz = compute_deformation_components(dense_flow)
        
        # Save magnitude
        nib.save(nib.Nifti1Image(magnitude, 
                                nib.load(img_fix_path).affine), 
                '{}_deformation_magnitude.nii.gz'.format(output_path))
        
        # Save components
        nib.save(nib.Nifti1Image(dx, 
                                nib.load(img_fix_path).affine), 
                '{}_deformation_x.nii.gz'.format(output_path))
        
        nib.save(nib.Nifti1Image(dy, 
                                nib.load(img_fix_path).affine), 
                '{}_deformation_y.nii.gz'.format(output_path))
        
        nib.save(nib.Nifti1Image(dz, 
                                nib.load(img_fix_path).affine), 
                '{}_deformation_z.nii.gz'.format(output_path))
        
        print('  {}_deformation_magnitude.nii.gz'.format(output_path))
        print('  {}_deformation_x.nii.gz'.format(output_path))
        print('  {}_deformation_y.nii.gz'.format(output_path))
        print('  {}_deformation_z.nii.gz'.format(output_path))
    
    print('Total computation time: {:.1f} s'.format(t1-t0))
    print()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='corrField registration')
    
    parser.add_argument('-F', '--fixed', required=True,
                        help="fixed image (*.nii/*.nii.gz)")
    parser.add_argument('-M', '--moving', required=True,
                        help="moving image (*.nii/*.nii.gz)")
    parser.add_argument('-m', '--mask', required=True,
                        help="mask for fixed image (*.nii/*.nii.gz)")
    parser.add_argument('-A', '--ablation_mask', required=True,
                        help="binary mask of ablation zone in moving image (*.nii/*.nii.gz)")
    parser.add_argument('-O', '--output', required=True,
                        help="output name (no filename extension)")
    parser.add_argument('--save_deformation', action='store_true', default=False,
                        help="Generate and save deformation field files (default: False)")

    parser.add_argument('-a', '--alpha', default=2.5,
                        help="regularisation parameter (default: 2.5)")
    parser.add_argument('-b', '--beta', default=150,
                        help="intensity weighting (default: 150)")
    parser.add_argument('-g', '--gamma', default=5,
                        help="scaling factor for soft correspondeces (default: 5)")
    parser.add_argument('-d', '--delta', default=1,
                        help="step size for mind descriptor (default: 1)")
    parser.add_argument('-l', '--lambd', default=0,
                        help="regularistion parameter for TPS (default: 0)")
    parser.add_argument('-s', '--sigma', default=1.4,
                        help="sigma for foerstner operator (default: 1.4)")
    parser.add_argument('-s1', '--sigma1', default=1,
                        help="sigma for mind descriptor (default: 1)")
    
    # New parameters for ablation zone handling
    parser.add_argument('-bd', '--border_dist', default=10, type=int,
                        help="distance from ablation border to increase density (default: 10)")
    parser.add_argument('-bf', '--border_density', default=2.0, type=float,
                        help="factor to increase keypoint density near borders (default: 2.0)")
    
    parser.add_argument('-L', '--search_radius', default="16x8",
                        help="maximum search radius for each level (default: 16x8)")
    parser.add_argument('-N', '--length', default="6x3",
                        help="cube-length of non-maximum suppression (default: 6x3)")
    parser.add_argument('-Q', '--quantisation', default="2x1",
                        help="quantisation of search step size (default: 2x1)")
    parser.add_argument('-R', '--patch_radius', default="3x2",
                        help="patch radius for similarity seach (default: 3x2)")
    parser.add_argument('-T', '--transform', default="nxn",
                        help="rigid(r)/non-rigid(n) (default: nxn)")
    
    args = parser.parse_args()
    main(args)