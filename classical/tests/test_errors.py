from toxic_clf.errors import categorize


def test_categories():
    c = categorize("Да пошёл ты на хуй", oov_share=0.0)
    assert c["short"] and c["obscene"] and not c["no_explicit_markers"]
    c = categorize("Уважаемый, вы, кажется, не вполне понимаете, о чём пишете в этом треде", oov_share=0.0)
    assert not c["short"] and c["no_explicit_markers"]
    assert categorize("ну ты д.ебил", oov_share=0.0)["obfuscation_or_typos"]
    assert categorize("какды чейто", oov_share=1.0)["obfuscation_or_typos"]
    assert not categorize("Небольшая заметка про погоду в городе", oov_share=0.0)["obscene"]
