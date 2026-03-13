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



def test_truncate_text_short_input() -> None:
    """Test truncate_text returns short text unchanged."""
    from graphrag_studio.utils import truncate_text
    
    short_text = "This is a short text."
    assert truncate_text(short_text) == short_text



def test_sentence_split() -> None:
    """Test sentence_split correctly splits text."""
    from graphrag_studio.utils import sentence_split
    
    text = "First sentence. Second sentence! Third?"
    sentences = sentence_split(text)
    assert len(sentences) == 3
    assert sentences[0] == "First sentence."



def test_heuristic_entity_candidates_empty_text() -> None:
    """Test heuristic_entity_candidates handles empty text."""
    entities = heuristic_entity_candidates("")
    assert entities == []



def test_normalize_key_with_special_characters() -> None:
    """Test normalize_key handles special characters correctly."""
    assert normalize_key("Orion-Gateway_v2.0") == "orion-gateway-v2-0"
