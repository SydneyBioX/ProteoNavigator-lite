from proteonavigator_lite.agent import SYSTEM_PROMPT


def test_onboarding_guides_instead_of_only_listing_stage_names():
    prompt = " ".join(SYSTEM_PROMPT.split())
    assert "do not dump a menu" in prompt
    assert "Recommend a sensible default" in prompt
    assert "inspect/select core" in prompt


def test_rf_rescue_is_explained_with_its_tradeoff():
    assert "attempts to label" in SYSTEM_PROMPT
    assert "only cells the hierarchy left Unassigned" in SYSTEM_PROMPT
    assert "pseudo-label/error-propagation risk" in SYSTEM_PROMPT
