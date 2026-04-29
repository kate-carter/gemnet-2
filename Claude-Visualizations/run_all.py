#!/usr/bin/env python3
"""Run all visualization scripts in sequence."""

import sys, os, time
sys.path.insert(0, os.path.dirname(__file__))
os.chdir(os.path.dirname(__file__))

import matplotlib
matplotlib.use("Agg")

scripts = [
    ("Correlation landscape (all methods)",    "vis_correlation_landscape"),
    ("SemD network integration (all methods)", "vis_semd_network"),
    ("Human vs LLM comparison",                "vis_human_comparison"),
    ("Cross-method comparison",                "vis_method_comparison"),
    ("Human correlation by method",            "vis_human_correlation"),
    ("Human alignment (all methods)",          "vis_human_alignment"),
    ("Structural drivers (all methods)",       "vis_structural_drivers"),
    ("SemD predictors summary",                "vis_semd_predictors"),
    ("Human alignment predictors",             "vis_human_predictors"),
    ("Network gallery",                        "vis_network_gallery"),
]

for label, module_name in scripts:
    print(f"\n{'='*60}")
    print(f"  {label}")
    print(f"{'='*60}")
    t0 = time.time()
    try:
        mod = __import__(module_name)

        # Multi-method wrappers (preferred when present)
        if hasattr(mod, "plot_all_methods_correlation_landscape"):
            mod.plot_all_methods_correlation_landscape()
        if hasattr(mod, "plot_all_methods_erank_heatmap"):
            mod.plot_all_methods_erank_heatmap()
        if hasattr(mod, "plot_all_methods_semd_network"):
            mod.plot_all_methods_semd_network()
        if hasattr(mod, "plot_all_methods_human_alignment"):
            mod.plot_all_methods_human_alignment()
        if hasattr(mod, "plot_structural_drivers"):
            mod.plot_structural_drivers()
        if hasattr(mod, "plot_semd_predictors"):
            mod.plot_semd_predictors()
        if hasattr(mod, "plot_human_predictors"):
            mod.plot_human_predictors()

        # Single-call functions
        if hasattr(mod, "plot_human_comparison"):
            mod.plot_human_comparison()
        if hasattr(mod, "plot_metric_profiles"):
            mod.plot_metric_profiles()
        if hasattr(mod, "plot_llm_network_gallery"):
            mod.plot_llm_network_gallery()
        if hasattr(mod, "plot_human_network_gallery"):
            mod.plot_human_network_gallery()
        if hasattr(mod, "plot_side_by_side"):
            mod.plot_side_by_side()
        if hasattr(mod, "run_all") and module_name == "vis_method_comparison":
            mod.run_all()
        if hasattr(mod, "plot_human_correlation"):
            mod.plot_human_correlation()

        dt = time.time() - t0
        print(f"  ✓ {label} completed in {dt:.1f}s")
    except Exception as e:
        dt = time.time() - t0
        print(f"  ✗ {label} failed after {dt:.1f}s: {e}")
        import traceback
        traceback.print_exc()

print(f"\n{'='*60}")
print("  All done! Check vis_results/ for generated figures.")
print(f"{'='*60}")
