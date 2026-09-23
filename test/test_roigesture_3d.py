#!/usr/bin/env python3
"""
Test script to simulate the roigesture_3D flow and debug dimension mismatches.
"""
import numpy as np

def reshape_x_pos_old(arr: np.ndarray, pos, cfg) -> tuple:
    """Old version with the bug"""
    if cfg.mode == "depth":
        if arr.ndim == 4 and arr.shape[1] == 2:
            arr = arr.sum(axis=0)
        if arr.ndim == 2:
            arr = arr[..., None]
        elif arr.ndim == 3 and arr.shape[0] not in (1, 2, 3, 4):
            arr = np.transpose(arr, (1, 2, 0))
        if pos is not None and hasattr(pos, "ndim") and pos.ndim == 4:  # BUG!
            if pos.ndim == 2:  # This can never be true when pos.ndim == 4
                pos = pos[..., None]
            elif pos.ndim == 3 and pos.shape[0] not in (1, 2, 3, 4):
                pos = np.transpose(pos, (1, 2, 0))
        return arr, pos

def reshape_x_pos_new(arr: np.ndarray, pos, cfg) -> tuple:
    """New fixed version - properly handles [T, C, H, W] by collapsing T and moving C to last dim"""
    if cfg.mode == "depth":
        if arr.ndim == 4:
            # Sum over time (axis 0) to collapse T: [T, C, H, W] -> [C, H, W]
            arr = arr.sum(axis=0)  # [C, H, W]
        
        # Now arr is either [C, H, W] or already [H, W] or [H, W, C]
        if arr.ndim == 3 and arr.shape[0] in (1, 2, 3, 4):
            # This is [C, H, W] where C is small (1-4 channels) - move to last dim
            arr = np.transpose(arr, (1, 2, 0))  # [H, W, C]
        elif arr.ndim == 2:
            # This is [H, W], add channel dimension
            arr = arr[..., None]  # [H, W, 1]
        elif arr.ndim == 3 and arr.shape[0] not in (1, 2, 3, 4):
            # This is [T, H, W] where T > 4 - move T to last  
            arr = np.transpose(arr, (1, 2, 0))  # [H, W, T]
        
        if pos is not None and hasattr(pos, "ndim"):
            # Apply same transformations to pos as to arr
            if pos.ndim == 4:
                # Sum over time: [T, C, H, W] -> [C, H, W]
                pos = pos.sum(axis=0)
            
            if pos.ndim == 3 and pos.shape[0] in (1, 2, 3, 4):
                # [C, H, W] -> [H, W, C]
                pos = np.transpose(pos, (1, 2, 0))
            elif pos.ndim == 2:
                # [H, W] -> [H, W, 1]
                pos = pos[..., None]
            elif pos.ndim == 3 and pos.shape[0] not in (1, 2, 3, 4):
                # [T, H, W] -> [H, W, T] where T > 4
                pos = np.transpose(pos, (1, 2, 0))
        return arr, pos


class MockConfig:
    def __init__(self):
        self.mode = "depth"
        self.dataset = "roigesture_3D"
        self.channels = 4

cfg = MockConfig()

print("=" * 70)
print("Testing roigesture_3D flow with different pos shapes")
print("=" * 70)

# Test case 1: pos with shape [T, 2, H, W] (most common)
print("\nTest 1: pos.shape = (16, 2, 32, 32) [T, 2, H, W]")
print("-" * 70)
arr = np.random.rand(16, 2, 32, 32)
pos = np.random.rand(16, 2, 32, 32)

# Simulate the dataset_to_numpy flow
print("  Step 1: reshape_x_pos(arr, pos, cfg) for x_reshaped")
x_reshaped, _ = reshape_x_pos_new(arr, pos, cfg)
print(f"    x_reshaped.shape = {x_reshaped.shape}")

print("  Step 2: reshape_x_pos(pos, None, cfg) for pos_reshaped")
pos_array = np.array(pos)
print(f"    pos raw.shape = {pos_array.shape}")
pos_reshaped, _ = reshape_x_pos_new(pos_array, None, cfg)
print(f"    pos_reshaped.shape = {pos_reshaped.shape}")

print("  Step 3: concatenate")
try:
    x_combined = np.concatenate([x_reshaped, pos_reshaped], axis=-1)
    print(f"    ✓ Success! x_combined.shape = {x_combined.shape}")
except ValueError as e:
    print(f"    ✗ Failed: {e}")

# Test case 2: pos with shape [T, 1, H, W] (single polarity)
print("\nTest 2: pos.shape = (16, 1, 32, 32) [T, 1, H, W]")
print("-" * 70)
arr = np.random.rand(16, 2, 32, 32)
pos = np.random.rand(16, 1, 32, 32)

print("  Step 1: reshape_x_pos(arr, pos, cfg)")
x_reshaped, _ = reshape_x_pos_new(arr, pos, cfg)
print(f"    x_reshaped.shape = {x_reshaped.shape}")

print("  Step 2: reshape_x_pos(pos, None, cfg)")
pos_array = np.array(pos)
print(f"    pos raw.shape = {pos_array.shape}")
pos_reshaped, _ = reshape_x_pos_new(pos_array, None, cfg)
print(f"    pos_reshaped.shape = {pos_reshaped.shape}")

print("  Step 3: concatenate")
try:
    x_combined = np.concatenate([x_reshaped, pos_reshaped], axis=-1)
    print(f"    ✓ Success! x_combined.shape = {x_combined.shape}")
except ValueError as e:
    print(f"    ✗ Failed: {e}")

# Test case 3: pos with shape [T, H, W] (no polarity dimension)
print("\nTest 3: pos.shape = (16, 32, 32) [T, H, W] (no polarity dim)")
print("-" * 70)
arr = np.random.rand(16, 2, 32, 32)
pos = np.random.rand(16, 32, 32)

print("  Step 1: reshape_x_pos(arr, pos, cfg)")
x_reshaped, _ = reshape_x_pos_new(arr, pos, cfg)
print(f"    x_reshaped.shape = {x_reshaped.shape}")

print("  Step 2: reshape_x_pos(pos, None, cfg)")
pos_array = np.array(pos)
print(f"    pos raw.shape = {pos_array.shape}")
pos_reshaped, _ = reshape_x_pos_new(pos_array, None, cfg)
print(f"    pos_reshaped.shape = {pos_reshaped.shape}")

print("  Step 3: concatenate")
try:
    x_combined = np.concatenate([x_reshaped, pos_reshaped], axis=-1)
    print(f"    ✓ Success! x_combined.shape = {x_combined.shape}")
except ValueError as e:
    print(f"    ✗ Failed: {e}")

# Compare OLD vs NEW code for problematic case
print("\n" + "=" * 70)
print("Comparison: OLD (buggy) vs NEW (fixed) for [T, 2, H, W]")
print("=" * 70)

arr = np.random.rand(16, 2, 32, 32)
pos = np.random.rand(16, 2, 32, 32)

print("\nOLD CODE:")
x_old, _ = reshape_x_pos_old(arr, np.array(pos), cfg)
pos_old, _ = reshape_x_pos_old(np.array(pos), None, cfg)
print(f"  x_reshaped.shape = {x_old.shape}, pos_reshaped.shape = {pos_old.shape}")
try:
    x_combined_old = np.concatenate([x_old, pos_old], axis=-1)
    print(f"  Concatenation: ✓ {x_combined_old.shape}")
except ValueError as e:
    print(f"  Concatenation: ✗ {e}")

print("\nNEW CODE:")
x_new, _ = reshape_x_pos_new(arr, np.array(pos), cfg)
pos_new, _ = reshape_x_pos_new(np.array(pos), None, cfg)
print(f"  x_reshaped.shape = {x_new.shape}, pos_reshaped.shape = {pos_new.shape}")
try:
    x_combined_new = np.concatenate([x_new, pos_new], axis=-1)
    print(f"  Concatenation: ✓ {x_combined_new.shape}")
except ValueError as e:
    print(f"  Concatenation: ✗ {e}")

print("\n" + "=" * 70)
print("✓ Testing complete")
print("=" * 70)
