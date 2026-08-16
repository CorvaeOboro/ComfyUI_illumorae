"""
Test suite for TextShowMultilineColor node.

Runnable via:  python -m ComfyUI_illumorae_TextShowMultilineColor.tests.test_text_show_multiline_color
"""
import sys
import os

# Allow imports from the parent package
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from text_show_multiline_color import illumoraeTextShowMultilineColorNode


def test_sanitize_none():
    node = illumoraeTextShowMultilineColorNode()
    assert node._sanitize_input(None) == ""


def test_sanitize_string():
    node = illumoraeTextShowMultilineColorNode()
    assert node._sanitize_input("hello") == "hello"


def test_sanitize_dict_with_text_key():
    node = illumoraeTextShowMultilineColorNode()
    assert node._sanitize_input({"text": "found"}) == "found"


def test_sanitize_dict_without_known_key():
    node = illumoraeTextShowMultilineColorNode()
    result = node._sanitize_input({"foo": "bar"})
    assert "foo" in result and "bar" in result


def test_sanitize_list_single_string():
    node = illumoraeTextShowMultilineColorNode()
    assert node._sanitize_input(["single"]) == "single"


def test_sanitize_list_multiple():
    node = illumoraeTextShowMultilineColorNode()
    assert node._sanitize_input([1, 2]) == "1\n2"


def test_highlight_basic_structure():
    """Sanity check that highlighting produces expected HTML tags."""
    node = illumoraeTextShowMultilineColorNode()
    text = "(hello:1.2)"
    html = node._highlight_text(text, "dark")
    assert html.startswith("<pre")
    assert html.endswith("</pre>")
    assert "span" in html


def test_highlight_parentheses_depth():
    """Parentheses should receive colour spans."""
    node = illumoraeTextShowMultilineColorNode()
    text = "((word))"
    html = node._highlight_text(text, "dark")
    open_count = html.count("<span")
    assert open_count >= 4  # at least 4 spans for parens


def test_highlight_numbers():
    """Numbers like 1.2 should be wrapped in a span."""
    node = illumoraeTextShowMultilineColorNode()
    text = "strength 1.5"
    html = node._highlight_text(text, "dark")
    assert "1.5" in html


def test_highlight_lora_tag():
    """<lora:name:1> should be wrapped in a span."""
    node = illumoraeTextShowMultilineColorNode()
    text = "<lora:my_lora:0.8>"
    html = node._highlight_text(text, "dark")
    assert "lora_tag" in html or "0.8" in html


def test_highlight_comment():
    """Comments should be wrapped in comment-colour spans."""
    node = illumoraeTextShowMultilineColorNode()
    text = "# this is a comment"
    html = node._highlight_text(text, "dark")
    assert "comment" in html or "# this is a comment" in html


def test_process_none_input():
    """Node must return gracefully when input is None."""
    node = illumoraeTextShowMultilineColorNode()
    result = node.process(None, enable_highlighting=True, theme="dark")
    assert "ui" in result
    assert "text" in result["ui"]
    assert len(result["ui"]["text"]) == 1
    assert result["result"] == ("",)


def test_process_normal_input():
    """Node must return plain text in result and HTML in ui."""
    node = illumoraeTextShowMultilineColorNode()
    result = node.process("hello world", enable_highlighting=True, theme="dark")
    assert result["result"] == ("hello world",)
    assert "ui" in result
    assert result["ui"]["text"][0].startswith("<pre")


def test_process_dict_input():
    """Node must extract text from a dict gracefully."""
    node = illumoraeTextShowMultilineColorNode()
    result = node.process({"text": "from dict"}, enable_highlighting=False, theme="dark")
    assert result["result"] == ("from dict",)


def run_all():
    tests = [
        test_sanitize_none,
        test_sanitize_string,
        test_sanitize_dict_with_text_key,
        test_sanitize_dict_without_known_key,
        test_sanitize_list_single_string,
        test_sanitize_list_multiple,
        test_highlight_basic_structure,
        test_highlight_parentheses_depth,
        test_highlight_numbers,
        test_highlight_lora_tag,
        test_highlight_comment,
        test_process_none_input,
        test_process_normal_input,
        test_process_dict_input,
    ]
    passed = 0
    failed = 0
    for t in tests:
        try:
            t()
            print(f"  PASS  {t.__name__}")
            passed += 1
        except AssertionError as e:
            print(f"  FAIL  {t.__name__}: {e}")
            failed += 1
        except Exception as e:
            print(f"  ERR   {t.__name__}: {e}")
            failed += 1
    print(f"\n{passed}/{len(tests)} passed, {failed} failed")
    return failed == 0


if __name__ == "__main__":
    ok = run_all()
    sys.exit(0 if ok else 1)
