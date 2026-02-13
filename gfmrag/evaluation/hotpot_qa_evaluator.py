# Adapt from: https://github.com/OSU-NLP-Group/HippoRAG/blob/main/src/qa/hotpotqa_evaluation.py
import logging
import re
import string
from collections import Counter

from gfmrag.evaluation.base_evaluator import BaseEvaluator

logger = logging.getLogger(__name__)


def normalize_answer(s: str) -> str:
    def remove_articles(text: str) -> str:
        return re.sub(r"\b(a|an|the)\b", " ", text)

    def white_space_fix(text: str) -> str:
        return " ".join(text.split())

    def remove_punc(text: str) -> str:
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)

    def lower(text: str) -> str:
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))


def f1_score(prediction: str, ground_truth: str) -> tuple:
    normalized_prediction = normalize_answer(prediction)
    normalized_ground_truth = normalize_answer(ground_truth)

    zero_metric = (0, 0, 0)

    if (
        normalized_prediction in ["yes", "no", "noanswer"]
        and normalized_prediction != normalized_ground_truth
    ):
        return zero_metric
    if (
        normalized_ground_truth in ["yes", "no", "noanswer"]
        and normalized_prediction != normalized_ground_truth
    ):
        return zero_metric

    prediction_tokens = normalized_prediction.split()
    ground_truth_tokens = normalized_ground_truth.split()
    common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return zero_metric
    precision = 1.0 * num_same / len(prediction_tokens)
    recall = 1.0 * num_same / len(ground_truth_tokens)
    f1 = (2 * precision * recall) / (precision + recall)
    return f1, precision, recall


def exact_match_score(prediction: str, ground_truth: str) -> int:
    return 1 if normalize_answer(prediction) == normalize_answer(ground_truth) else 0


def update_answer(metrics: dict, prediction: str, gold: str) -> tuple:
    em = exact_match_score(prediction, gold)
    f1, precision, recall = f1_score(prediction, gold)
    metrics["em"] += float(em)
    metrics["f1"] += f1
    metrics["precision"] += precision
    metrics["recall"] += recall
    return em, f1, precision, recall


class HotpotQAEvaluator(BaseEvaluator):
    """
    HotpotQAEvaluator
    """

    def __init__(self, prediction_file: str, log_per_query: bool = False) -> None:
        super().__init__(prediction_file)
        self.log_per_query = log_per_query

    @staticmethod
    def evaluate_single(pred: dict) -> dict:
        """Evaluate a single prediction and return its metrics."""
        if "Answer: " in pred["response"]:
            pre_ans = pred["response"].split("Answer:")[1].strip()
        else:
            pre_ans = pred["response"]
        
        gold = pred["answer"]
        em = exact_match_score(pre_ans, gold)
        f1, precision, recall = f1_score(pre_ans, gold)
        
        return {
            "em": float(em),
            "f1": f1,
            "precision": precision,
            "recall": recall,
        }

    def evaluate(self) -> dict:
        metrics = {"em": 0.0, "f1": 0.0, "precision": 0.0, "recall": 0.0}

        for pred in self.data:
            single_metrics = self.evaluate_single(pred)
            
            if self.log_per_query:
                logger.info(
                    f"Query id={pred.get('id', 'N/A')}: "
                    f"em={single_metrics['em']:.4f}, f1={single_metrics['f1']:.4f}, "
                    f"precision={single_metrics['precision']:.4f}, recall={single_metrics['recall']:.4f}"
                )
            
            for k in metrics.keys():
                metrics[k] += single_metrics[k]

        n = len(self.data)
        for k in metrics.keys():
            metrics[k] /= n
        return metrics
