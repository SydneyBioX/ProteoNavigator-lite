from proteonavigator_lite.services import requested_tree_cell_type


def test_resolves_cell_type_from_natural_tree_plot_request():
    available = ["BnTcell", "Myeloid", "Tumor"]
    assert (
        requested_tree_cell_type("Could you show me the tree plot for Tumor markers?", available)
        == "Tumor"
    )


def test_does_not_intercept_unrelated_tree_discussion():
    assert requested_tree_cell_type("What does tree gating do?", ["Tumor"]) is None
