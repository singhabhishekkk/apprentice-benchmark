from prepare_data import CLASS_NAMES, MAX_CONTENT_CHARS, build_input


def test_build_input_fills_paperless_prompt_and_caps_content() -> None:
    text = "x" * (MAX_CONTENT_CHARS + 10)
    prompt = build_input(text)

    assert "<available_document_types>\n" + ", ".join(CLASS_NAMES) in prompt
    assert "<title>\n</title>" in prompt
    assert ("x" * MAX_CONTENT_CHARS) in prompt
    assert ("x" * (MAX_CONTENT_CHARS + 1)) not in prompt
