# ProteoNavigator Lite

A local-first LLM agent for building, reviewing, and running cell-annotation
workflows with the Python **cytoGater** package.

Upload AnnData (`.h5ad`), a cells-by-markers CSV, or an R
SpatialExperiment/SingleCellExperiment (`.rds`). You may also upload a lineage
CSV, or ask the agent to propose a lineage table from the available panel.

Every lineage proposal is validated against the real marker panel. Annotation
runs require an explicit plan approval.
