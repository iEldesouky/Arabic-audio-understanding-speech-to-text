from audio_understanding.advanced.keyword_spotting import KeywordSpotter


def test_keyword_spotting_counts_keyword_occurrences() -> None:
    spotter = KeywordSpotter()
    transcript = "هذا امتحان مهم وهذا موعد الامتحان النهائي"
    hits = spotter.find_keywords(transcript, ["امتحان", "موعد", "غيرموجود"])

    as_dict = {h.keyword: h.count for h in hits}
    assert as_dict["امتحان"] >= 1
    assert as_dict["موعد"] == 1
    assert "غيرموجود" not in as_dict
