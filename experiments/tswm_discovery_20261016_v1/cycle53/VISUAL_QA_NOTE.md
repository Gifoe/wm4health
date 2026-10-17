# Final figure layout

Final PNG/SVG figures use render_native53_v5.py after visual inspection found cropped contrast labels and vertical-label overlap in the previous fixed layout. Contrast labels use obs (observation concat), concat (frozen-AE concat), Gate (frozen-AE Gate) and Gate + DS (directional supervision). Panel spacing and axis-label clearance were increased. Values, seed SDs, block intervals and series are unchanged.

All 84 retained forecast groups and seven additional numerical files were hash-checked unchanged during this revision. Earlier native layout/PDF failures and all partial figures remain on the server. The final export uses PNG/SVG, with no PDF claim. Qualification is in outputs/plot_layout_revision_v1/qualification.json.
