"""Preserve completed diagnostic sections when earlier reports are regenerated."""
def diagnostics_fragment(root):
    path=root/'docs/PAPER_DIAGNOSTICS_RESULTS.md'
    if not path.exists():return ''
    return path.read_text().replace('](../results/','](results/').replace('](../configs/','](configs/').replace('](figures/','](docs/figures/').replace('](PAPER_STAGE2_DESIGN.md)','](docs/PAPER_STAGE2_DESIGN.md)')
