import json
from pathlib import Path

from proteonavigator_lite.models import SessionWorkspace
from proteonavigator_lite.services import WorkspaceService


def test_service_renders_selected_tree_plot(tmp_path: Path):
    tree = {
        "type": "node",
        "depth": 0,
        "cells": list(range(6)),
        "marker": "CD3",
        "cutoff": 0.5,
        "sep_score": 0.8,
        "left": {"type": "leaf", "depth": 1, "cells": [0, 1, 2]},
        "right": {"type": "leaf", "depth": 1, "cells": [3, 4, 5]},
    }
    output_dir = tmp_path / "outputs"
    output_dir.mkdir()
    tree_path = output_dir / "trees.json"
    tree_path.write_text(json.dumps({"Tumor": tree}))
    service = WorkspaceService(SessionWorkspace(root=tmp_path))
    service.workspace.last_result = {
        "output_path": str(output_dir / "annotated.h5ad"),
        "tree_path": str(tree_path),
        "tree_cell_types": ["Tumor"],
    }

    result = service.create_tree_plot("tumor")

    assert result["cell_type"] == "Tumor"
    assert Path(result["path"]).exists()
    assert service.consume_visualizations() == [result]
    assert service.consume_visualizations() == []
