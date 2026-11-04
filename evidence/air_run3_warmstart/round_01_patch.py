# Final edit: original weights plus 2.4x the rank-1 anti-repetition delta in two down_proj tensors
for l in (15, 19):
    n = f"model.layers.{l}.mlp.down_proj.weight"
    save_edit(n, load_original(n) + 2.4 * load_artifact(f"delta_{l}"))
