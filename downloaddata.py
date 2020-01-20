from pytest import mark

import base


class TestBilstmTagger(base.PerformanceBase):

    @mark.parametrize("language, syntax_path, mentions_path",
                      [("ar", "syntax_izumo_ar_stock", "entity-mentions_bilstm_ar_stock"),
                       ("de", "syntax_izumo_de_stock", "entity-mentions_bilstm_de_stock"),
                       ("en", "syntax_izumo_en_stock", "entity-mentions_bilstm_en_stock"),
                       ("es", "syntax_izumo_es_stock", "entity-mentions_bilstm_es_stock"),
                       ("fr", "syntax_izumo_fr_stock", "entity-mentions_bilstm_fr_stock"),
                       ("it", "syntax_izumo_it_stock", "entity-mentions_bilstm_it_stock"),
                       ("ja", "syntax_izumo_ja_stock", "entity-mentions_bilstm_ja_stock"),
                       ("ko", "syntax_izumo_ko_stock", "entity-mentions_bilstm_ko_stock"),
                       ("nl", "syntax_izumo_nl_stock", "entity-mentions_bilstm_nl_stock"),
                       ("pt", "syntax_izumo_pt_stock", "entity-mentions_bilstm_pt_stock"),
                       ("zh_s", "syntax_izumo_zh-cn_stock", "entity-mentions_bilstm_zh-cn_stock")])
    def test_perf_pipeline(self, language, syntax_path, mentions_path):
        data_set = self.pull_test_data(language)
        for benchmark in data_set:
            syntax_model = self.get_model(syntax_path, language)
            mentions_model = self.get_model(mentions_path, language)
            self.block_performance(benchmark, language, (mentions_model, lambda raw_doc: syntax_model.run(raw_doc)))