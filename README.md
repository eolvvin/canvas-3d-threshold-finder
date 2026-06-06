# Canvas 3+1D Threshold Finder

GPU-accelerated 3+1D simulation using Taichi. Searches for the critical threshold R where bound states form in three spatial dimensions plus time. Uses locked weights derived from first principles.

## What It Verifies

The canvas model predicts R_ST = d+1 = 4 as the spacetime threshold. This program performs a self-adjusting binary search in full 3+1D to find the critical R and verifies bound state formation at the predicted value.

## Requirements

- Python 3.8+
- Taichi 1.6+
- NumPy, Matplotlib

## Quick Start

```

pip install taichi numpy matplotlib
python canvas_3d_threshold_finder.py

```

## Simulation Parameters

| Parameter | Value |
|-----------|-------|
| Grid | 48³ = 110,592 cells |
| Time steps | 1500 |
| dt | 0.002 |
| Weights | Locked to c_eff/d_eff = π/2 |
| GPU | Auto-detected |

## How It Works

1. Tests at predicted R = 4.0 with increasing amplitude
2. Binary search to find precise critical R
3. Verification with 5 independent runs
4. Auto-adjusts amplitude if needed

## Results

- Critical R from 3+1D binary search
- Verification success rate at R = 4.0
- Locked weights preserved throughout

## Hardware Notes

- Tested on Radeon RX 560 (4GB VRAM)
- Minimum 2GB VRAM recommended
- Falls back to CPU if no GPU available

## Citation

Ong, E. A Unified Framework of Fundamental Physics (2026).

## License

MIT
