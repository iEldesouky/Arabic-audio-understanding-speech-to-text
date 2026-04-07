from audio_understanding.advanced.keyword_spotting import KeywordSpotter


def test_keyword_spotting_normalization() -> None:
    spotter = KeywordSpotter()
    transcript = "هذا موعد الامتحان النهائي والطوارئ مهمة"
    keywords = ["موعد", "امتحان", "طوارئ", "غير موجود"]

    hits = spotter.find_keywords(transcript, keywords)
    hit_map = {h.keyword: h.count for h in hits}

    assert hit_map["موعد"] == 1
    assert hit_map["امتحان"] == 1
    assert hit_map["طوارئ"] == 1
    assert "غير موجود" not in hit_map
