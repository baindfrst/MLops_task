from src.api.rag_app import llm_answer

def test_prompt_structure(monkeypatch):
    class Dummy:
        def generate(self, **kwargs):
            return [[1, 2, 3]]

    class DummyTok:
        def __call__(self, text, return_tensors):
            assert "ТЕСТОВЫЙ_КОНТЕКСТ" in text
            class DummyInput(dict):
                def to(self, *args, **kwargs): return self
            return DummyInput({"input_ids": [[1]]})
            
        def decode(self, ids, skip_special_tokens):
            return "ok"

    monkeypatch.setattr("src.api.rag_app.gen_model", Dummy())
    monkeypatch.setattr("src.api.rag_app.gen_tok", DummyTok())

    answer = llm_answer("тестовый вопрос", "ТЕСТОВЫЙ_КОНТЕКСТ")
    assert answer == "ok"