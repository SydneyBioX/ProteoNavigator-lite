from proteonavigator_lite.knowledge import describe_options, inspect_public_api
from proteonavigator_lite.services import STAGE_DEFAULTS


def test_cutoff_method_options_are_explicit():
    result = describe_options(STAGE_DEFAULTS, "tree_gating", "cutoff_method")

    assert result["default"] == "mean"
    assert result["choices"] == ["mean", "equal_posteriors"]


def test_live_api_inspection_reports_signature():
    result = inspect_public_api("run_tree_gating")

    assert "cutoff_method" in result["signature"]
    assert result["module"].startswith("cytogater")
