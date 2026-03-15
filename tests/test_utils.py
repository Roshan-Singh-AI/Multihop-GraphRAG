from graphrag_studio.utils import chunk_ref, heuristic_entity_candidates, normalize_key



def test_normalize_key() -> None:
    assert normalize_key("Orion Gateway") == "orion-gateway"



def test_chunk_ref() -> None:
    assert chunk_ref("atlas-doc", 3) == "atlas-doc::chunk_003"



def test_heuristic_entity_candidates() -> None:
    text = "Atlas Plant uses Orion Gateway and WeldVision Camera."
    entities = heuristic_entity_candidates(text)
    assert "Atlas Plant" in entities
    assert "Orion Gateway" in entities



def test_normalize_key_with_special_characters() -> None:
    """Test normalize_key handles special characters correctly."""
    assert normalize_key("Orion-Gateway_v2.0") == "orion-gateway-v2-0"
    assert normalize_key("Atlas   Plant") == "atlas-plant"
