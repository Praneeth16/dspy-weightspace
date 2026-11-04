# Final: published sparse override on layers 15 and 19 down_proj, delta scaled x1.9
names = ["model.layers.15.mlp.down_proj.weight", "model.layers.19.mlp.down_proj.weight"]
for n in names:
    o = load_original(n)
    d = load_artifact(n.replace(".", "_") + "_d1")  # saved published delta (1.0x)
    save_edit(n, o + 1.9 * d)
