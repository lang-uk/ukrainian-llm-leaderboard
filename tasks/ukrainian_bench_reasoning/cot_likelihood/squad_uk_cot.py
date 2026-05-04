import math
import sys
from functools import partial
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
TASK_DIR = Path(__file__).resolve().parent
if str(TASK_DIR) not in sys.path:
    sys.path.insert(0, str(TASK_DIR))

from lm_eval.api.instance import Instance
from lm_eval.api.task import ConfigurableTask
from tasks.ukrainian_bench.squad_uk.task import _squad_agg

import utils


class SQuAD2CotLikelihood(ConfigurableTask):
    VERSION = "0.1"
    NO_ANSWER = "немає"
    HAS_ANSWER = "є"

    def __init__(self, config=None):
        config = dict(config or {})
        config.pop("class", None)
        super().__init__(config=config)

    def doc_to_text(self, doc, doc_to_text=None):
        return utils.squad_doc_to_text_cot(doc)

    def doc_to_target(self, doc, doc_to_target=None):
        return self.NO_ANSWER if len(doc["Answer"]["text"]) == 0 else self.HAS_ANSWER

    def construct_requests(self, doc, ctx, **kwargs):
        kwargs.pop("apply_chat_template", None)
        kwargs.pop("chat_template", None)
        return [
            Instance(
                request_type="loglikelihood",
                doc=doc,
                arguments=(ctx, self.NO_ANSWER),
                idx=0,
                **kwargs,
            ),
            Instance(
                request_type="loglikelihood",
                doc=doc,
                arguments=(ctx, self.HAS_ANSWER),
                idx=1,
                **kwargs,
            ),
        ]

    def process_results(self, doc, results):
        (logprob_no_answer, _), (logprob_has_answer, _) = results
        max_logprob = max(logprob_no_answer, logprob_has_answer)
        denom = max_logprob + math.log(
            math.exp(logprob_no_answer - max_logprob)
            + math.exp(logprob_has_answer - max_logprob)
        )
        no_answer_probability = math.exp(logprob_no_answer - denom)

        predictions = {
            "id": doc["id"],
            "prediction_text": utils.squad_prediction_from_trace(doc),
            "no_answer_probability": no_answer_probability,
        }
        references = {
            "id": doc["id"],
            "answers": doc["Answer"],
        }
        pair = (predictions, references)
        return {
            "exact": pair,
            "f1": pair,
            "HasAns_exact": pair,
            "HasAns_f1": pair,
            "NoAns_exact": pair,
            "NoAns_f1": pair,
            "best_exact": pair,
            "best_f1": pair,
        }

    def aggregation(self):
        return {
            "exact": partial(_squad_agg, "exact"),
            "f1": partial(_squad_agg, "f1"),
            "HasAns_exact": partial(_squad_agg, "HasAns_exact"),
            "HasAns_f1": partial(_squad_agg, "HasAns_f1"),
            "NoAns_exact": partial(_squad_agg, "NoAns_exact"),
            "NoAns_f1": partial(_squad_agg, "NoAns_f1"),
            "best_exact": partial(_squad_agg, "best_exact"),
            "best_f1": partial(_squad_agg, "best_f1"),
        }

    def higher_is_better(self):
        return {
            "exact": True,
            "f1": True,
            "HasAns_exact": True,
            "HasAns_f1": True,
            "NoAns_exact": True,
            "NoAns_f1": True,
            "best_exact": True,
            "best_f1": True,
        }
