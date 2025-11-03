from src.data.preprocess import clean_text

def test_clean_text_basic():
    text = "       тестовый | текст? для провеРки нормал!!   "
    cleaned = clean_text(text)
    print(cleaned)
    assert cleaned == "тестовый текст для проверки нормал"

def test_clean_text_empty():
    text = ""
    cleaned = clean_text(text)
    assert cleaned == ""
