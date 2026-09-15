#!/usr/bin/env Rscript

args <- commandArgs(trailingOnly = TRUE)
if (length(args) != 2) stop("Usage: convert_rds.R input.rds output_dir")

suppressPackageStartupMessages({
  library(SingleCellExperiment)
})

obj <- readRDS(args[[1]])
dir.create(args[[2]], recursive = TRUE, showWarnings = FALSE)
assay_name <- if ("exprs" %in% assayNames(obj)) "exprs" else assayNames(obj)[[1]]
expr <- t(as.matrix(assay(obj, assay_name)))
meta <- as.data.frame(colData(obj))
write.csv(expr, file.path(args[[2]], "expression.csv"), quote = FALSE)
write.csv(meta, file.path(args[[2]], "obs.csv"), quote = TRUE)

coords <- NULL
if (inherits(obj, "SpatialExperiment")) {
  suppressPackageStartupMessages(library(SpatialExperiment))
  coords <- spatialCoords(obj)
}

# Some SingleCellExperiment files keep coordinates as a reduced dimension.
if (is.null(coords)) {
  spatial_reduced_dim <- reducedDimNames(obj)[
    tolower(reducedDimNames(obj)) %in% c("spatial", "spatialcoords", "coordinates", "coords")
  ]
  if (length(spatial_reduced_dim) > 0) {
    coords <- reducedDim(obj, spatial_reduced_dim[[1]])
  }
}

# Fall back to common coordinate pairs in colData.
if (is.null(coords)) {
  coordinate_pairs <- list(
    c("x", "y"), c("X", "Y"), c("Pos_X", "Pos_Y"),
    c("center_x", "center_y"), c("CenterX", "CenterY")
  )
  for (pair in coordinate_pairs) {
    if (all(pair %in% colnames(meta))) {
      coords <- as.matrix(meta[, pair, drop = FALSE])
      break
    }
  }
}

if (!is.null(coords) && ncol(coords) >= 2) {
  rownames(coords) <- colnames(obj)
  write.csv(as.data.frame(coords[, 1:2, drop = FALSE]),
            file.path(args[[2]], "spatial.csv"), quote = FALSE)
}
