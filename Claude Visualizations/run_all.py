#!/usr/bin/env python3
"""Run all visualization scripts in sequence."""

import sys, os, time
sys.path.insert(0, os.path.dirname(__file__))
os.chdir(os.path.dirname(__file__))

import matplotlib
matplotlib.use("Agg")

scripts = [
    ("Threshold sweep",         "vis_threshold_sweep"),
    ("Correlation landscape",   "vis_correlation_landscape"),
    ("Human vs LLM comparison", "vis_human_comparison"),
    ("SemD integration",        "vis_semd_network"),
    ("Network gallery",         "vis_network_gallery"),
]

for label, module_name in scripts:
    print(f"\n{'='*60}")
    print(f"  {label}")
    print(f"{'='*60}")
    t0 = time.time()
    try:
        mod = __import__(module_name)
        # Each module's __main__ block calls the relevant functions;
        # replicate that here
        if hasattr(mod, "plot_threshold_sweep"):
            mod.plot_threshold_sweep()
        if hasattr(mod, "plot_correlation_landscape"):
            mod.plot_correlation_landscape()
        if hasattr(mod, "plot_concept_comparison_heatmap"):
            mod.plot_concept_comparison_heatmap()
        if hasattr(mod, "plot_human_comparison"):
            mod.plot_human_comparison()
        if hasattr(mod, "plot_metric_profiles"):
            mod.plot_metric_profiles()
        if hasattr(mod, "plot_semd_network"):
            mod.plot_semd_network()
        if hasattr(mod, "plot_llm_network_gallery"):
            mod.plot_llm_network_gallery()
        if hasattr(mod, "plot_human_network_gallery"):
            mod.plot_human_network_gallery()
        if hasattr(mod, "plot_side_by_side"):
            mod.plot_side_by_side()
        dt = time.time() - t0
        print(f"  ✓ {label} completed in {dt:.1f}s")
    except Exception as e:
        dt = time.time() - t0
        print(f"  ✗ {label} failed after {dt:.1f}s: {e}")
        import traceback
        traceback.print_exc()

print(f"\n{'='*60}")
print("  All done! Check output/ for generated figures.")
print(f"{'='*60}")
