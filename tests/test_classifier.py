from pipeline.classifier import classify


def test_financial_bucket():
    result = classify("I cannot afford my medication and utilities")
    assert "financial_assistance" in result.problem_classes
    assert "medical_care" in result.problem_classes


def test_special_case():
    result = classify("My neighbour keeps stealing my mail and I don't know who to call")
    assert result.problem_classes == []
    assert result.special_case is not None
    assert len(result.special_case) > 10


def test_multiple_buckets():
    result = classify("I am lonely and also need help with my FDW paperwork")
    assert "social_isolation" in result.problem_classes
    assert "caregiving_support" in result.problem_classes
