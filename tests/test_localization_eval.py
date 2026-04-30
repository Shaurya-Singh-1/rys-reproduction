from src.agent_eval.localization import parse_ranked_files, score_ranking, summarize_localization_records


def test_parse_ranked_files_from_json_completes_missing_candidates() -> None:
    candidates = ["a.py", "b.py", "c.py"]
    raw = '{"ranked_files": ["b.py", "a.py"], "rationale": "b looks relevant"}'
    ranked, meta = parse_ranked_files(raw, candidates)
    assert ranked == ["b.py", "a.py", "c.py"]
    assert meta["parser"] == "json"


def test_score_ranking_topk_and_mrr() -> None:
    score = score_ranking(["b.py", "a.py", "c.py"], "a.py")
    assert score["rank"] == 2
    assert score["top1"] is False
    assert score["top3"] is True
    assert score["mrr"] == 0.5


def test_summarize_localization_records_groups_by_condition() -> None:
    rows = [
        {"condition": "baseline", "score": {"top1": True, "top3": True, "mrr": 1.0, "rank": 1}},
        {"condition": "baseline", "score": {"top1": False, "top3": True, "mrr": 0.5, "rank": 2}},
    ]
    summary = summarize_localization_records(rows)
    assert summary == [
        {
            "condition": "baseline",
            "n": 2,
            "top1_accuracy": 0.5,
            "top3_accuracy": 1.0,
            "mrr": 0.75,
            "avg_rank": 1.5,
        }
    ]
