import random
from transformers import AutoModelForSequenceClassification, AutoTokenizer, AutoModelForSeq2SeqLM
from transformers import PreTrainedModel
import torch
from typing import List, Dict, Tuple, Any
from abc import abstractmethod
import numpy as np
from datasets import load_metric
from gem_metrics.msttr import MSTTR
from gem_metrics.ngrams import NGramStats
from rl4lms.envs.text_generation.caption_metrics.cider import Cider
from rl4lms.envs.text_generation.caption_metrics.spice.spice import Spice
from gem_metrics.texts import Predictions
from rl4lms.envs.text_generation.summ_metrics.summa_c import SummaCConv, SummaCZS
from rl4lms.data_pools.task_utils.totto.eval_utils import compute_parent, compute_bleu
from rl4lms.data_pools.custom_text_generation_pools import DailyDialog
from tqdm import tqdm
import copy
import rouge

class BaseMetric:
    @abstractmethod
    def compute(
        self,
        prompt_texts: List[str],
        generated_texts: List[str],
        reference_texts: List[List[str]],
        meta_infos: List[Dict[str, Any]] = None,
        model: PreTrainedModel = None,
        split_name: str = None,
    ):
        """
        Returns a dict where key is the metric name and value is again a dict consisting of tuple of individual scores (if any) and corpus level score

        eg. {
            metric_name: (individual_scores, corpus_level_score)
            "metric_1": ([0.5, 0.5, 0.8], 0.1)
        }

        """
        raise NotImplementedError


class LearnedRewardMetric(BaseMetric):
    def __init__(
        self,
        model_name: str,
        label_ix: int,
        batch_size: int,
        include_prompt_for_eval: bool = True,
    ) -> None:
        super().__init__()
        self._device = "cuda" if torch.cuda.is_available() else "cpu"
        self._tokenizer = AutoTokenizer.from_pretrained(model_name)
        self._tokenizer.truncation_side = "left"
        self._model = AutoModelForSequenceClassification.from_pretrained(model_name).to(
            self._device
        )
        self._label_ix = label_ix
        self._batch_size = batch_size
        self._include_prompt_for_eval = include_prompt_for_eval

    def compute(
        self,
        prompt_texts: List[str],
        generated_texts: List[str],
        reference_texts: List[List[str]],
        meta_infos: List[Dict[str, Any]] = None,
        model: PreTrainedModel = None,
        split_name: str = None,
    ) -> Dict[str, float]:
        all_scores = []
        current_ix = 0
        n_texts = len(generated_texts)
        while current_ix < n_texts:
            batch_gen_texts = generated_texts[
                current_ix : current_ix + self._batch_size
            ]
            batch_prompt_texts = prompt_texts[
                current_ix : current_ix + self._batch_size
            ]

            if self._include_prompt_for_eval:
                batch_gen_texts = [
                    (prompt + gen)
                    for gen, prompt in zip(batch_gen_texts, batch_prompt_texts)
                ]
            encoded = self._tokenizer(
                batch_gen_texts, return_tensors="pt", truncation=True, padding=True
            )
            with torch.no_grad():
                outputs = self._model(
                    input_ids=encoded.input_ids.to(self._device),
                    attention_mask=encoded.attention_mask.to(self._device),
                )
                scores = torch.softmax(outputs.logits, dim=1)
                scores = scores[:, self._label_ix].tolist()
                all_scores.extend(scores)
            current_ix += self._batch_size

        metric_dict = {
            "semantic/learned_automodel_metric": (all_scores, np.mean(all_scores))
        }
        return metric_dict


class MeteorMetric(BaseMetric):
    def __init__(self) -> None:
        super().__init__()
        self._metric = load_metric("meteor")

    def compute(
        self,
        prompt_texts: List[str],
        generated_texts: List[str],
        reference_texts: List[List[str]],
        meta_infos: List[Dict[str, Any]] = None,
        model: PreTrainedModel = None,
        split_name: str = None,
    ):

        score = self._metric.compute(
            predictions=generated_texts, references=reference_texts
        )["meteor"]

        metric_dict = {"lexical/meteor": (None, score)}
        return metric_dict


class RougeMetric(BaseMetric):
    def __init__(self, use_single_ref: bool = True) -> None:
        super().__init__()
        self._metric = load_metric("rouge")
        self._use_single_ref = use_single_ref

    def compute(
        self,
        prompt_texts: List[str],
        generated_texts: List[str],
        reference_texts: List[List[str]],
        meta_infos: List[Dict[str, Any]] = None,
        model: PreTrainedModel = None,
        split_name: str = None,
    ):
        if self._use_single_ref:
            # TBD: this is required for CNN/DM dataset, without this we get low scores
            # TBD: needs investigation
            ref_texts = [ref[0] for ref in reference_texts]
        else:
            ref_texts = reference_texts

        metric_results = self._metric.compute(
            predictions=generated_texts, references=ref_texts, use_stemmer=True
        )
        score_keys = ["rouge1", "rouge2", "rougeL", "rougeLsum"]
        metric_dict = {}
        for rouge_type in score_keys:
            rouge_score = metric_results[rouge_type].mid.fmeasure
            metric_dict[f"lexical/rouge_{rouge_type}"] = (None, rouge_score)
        return metric_dict


class BERTScoreMetric(BaseMetric):
    def __init__(self, language: str) -> None:
        super().__init__()
        self._metric = load_metric("bertscore")
        self._language = language
        # since models are loaded heavily on cuda:0, use the last one to avoid memory
        self._last_gpu = f"cuda:{torch.cuda.device_count() - 1}"

    def compute(
        self,
        prompt_texts: List[str],
        generated_texts: List[str],
        reference_texts: List[List[str]],
        meta_infos: List[Dict[str, Any]] = None,
        model: PreTrainedModel = None,
        split_name: str = None,
    ) -> Tuple[List[float], float]:
        with torch.no_grad():
            metric_results = self._metric.compute(
                predictions=generated_texts,
                references=reference_texts,
                lang=self._language,
                device=self._last_gpu,
            )
            bert_scores = metric_results["f1"]
            corpus_level_score = np.mean(bert_scores)
            metric_dict = {"semantic/bert_score": (bert_scores, corpus_level_score)}
            return metric_dict


class BLEUMetric(BaseMetric):
    def __init__(self) -> None:
        super().__init__()
        self._metric = load_metric("bleu")

    def compute(
        self,
        prompt_texts: List[str],
        generated_texts: List[str],
        reference_texts: List[List[str]],
        meta_infos: List[Dict[str, Any]] = None,
        model: PreTrainedModel = None,
        split_name: str = None,
    ) -> Tuple[List[float], float]:

        tokenized_predictions = []
        tokenized_reference_texts = []
        for prediction, refs in zip(generated_texts, reference_texts):
            tokenized_prediction = prediction.split()
            tokenized_refs = [ref.split() for ref in refs]
            tokenized_predictions.append(tokenized_prediction)
            tokenized_reference_texts.append(tokenized_refs)

        try:
            metric_results = self._metric.compute(
                predictions=tokenized_predictions, references=tokenized_reference_texts
            )
            bleu_score = metric_results["bleu"]
            metric_dict = {"lexical/bleu": (None, bleu_score)}
            return metric_dict
        except Exception as e:
            return {"lexical/bleu": (None, "n/a")}


class BLEURTMetric(BaseMetric):
    def __init__(self, config_name: str = None) -> None:
        super().__init__()
        self._metric = load_metric("bleurt", config_name=config_name)

    def compute(
        self,
        prompt_texts: List[str],
        generated_texts: List[str],
        reference_texts: List[List[str]],
        meta_infos: List[Dict[str, Any]] = None,
        model: PreTrainedModel = None,
        split_name: str = None,
    ) -> Tuple[List[float], float]:
        metric_results = self._metric.compute(
            predictions=generated_texts, references=reference_texts
        )
        corpus_score = np.mean(metric_results["scores"])
        metric_dict = {"semantic/bleurt": (metric_results["scores"], corpus_score)}
        return metric_dict


def get_generated_and_predictions(
    prompt_texts: List[str],
    generated_texts: List[str],
    reference_texts: List[List[str]],
    split_name: str,
):
    split_name = "" if split_name is None else split_name
    preds = {}
    refs = {}
    for ix, (prompt_text, gen_text, ref_text) in enumerate(
        zip(prompt_texts, generated_texts, reference_texts)
    ):
        preds[split_name + prompt_text] = [gen_text]
        refs[split_name + prompt_text] = ref_text
    return preds, refs


def get_individual_scores(
    prompt_texts: List[str], split_name: str, scores_dict: Dict[str, float]
):
    split_name = "" if split_name is None else split_name
    scores = []
    for prompt_text in prompt_texts:
        scores.append(scores_dict.get(split_name + prompt_text, "n/a"))
    return scores


class CIDERMetric(BaseMetric):
    def __init__(self) -> None:
        self._metric = Cider()

    def compute(
        self,
        prompt_texts: List[str],
        generated_texts: List[str],
        reference_texts: List[List[str]],
        meta_infos: List[Dict[str, Any]] = None,
        model: PreTrainedModel = None,
        split_name: str = None,
    ) -> Tuple[List[float], float]:
        predictions, references = get_generated_and_predictions(
            prompt_texts, generated_texts, reference_texts, split_name
        )
        (
            corpus_score,
            individual_scores,
        ) = self._metric.compute_score(references, predictions)
        individual_scores = get_individual_scores(
            prompt_texts, split_name, individual_scores
        )

        metric_dict = {"lexical/cider": (individual_scores, corpus_score)}
        return metric_dict


class SpiceMetric(BaseMetric):
    def __init__(self) -> None:
        self._metric = Spice()

    def compute(
        self,
        prompt_texts: List[str],
        generated_texts: List[str],
        reference_texts: List[List[str]],
        meta_infos: List[Dict[str, Any]] = None,
        model: PreTrainedModel = None,
        split_name: str = None,
    ) -> Tuple[List[float], float]:
        predictions, references = get_generated_and_predictions(
            prompt_texts, generated_texts, reference_texts, split_name
        )
        (
            corpus_score,
            individual_scores,
        ) = self._metric.compute_score(references, predictions)

        individual_scores = get_individual_scores(
            prompt_texts, split_name, individual_scores
        )

        metric_dict = {"lexical/spice": (individual_scores, corpus_score)}
        return metric_dict


class DiversityMetrics(BaseMetric):
    def __init__(self, window_size: int = 100) -> None:
        self._msttr_metric = MSTTR(window_size=window_size)
        self._n_gram_metric = NGramStats()

    def compute(
        self,
        prompt_texts: List[str],
        generated_texts: List[str],
        reference_texts: List[List[str]],
        meta_infos: List[Dict[str, Any]] = None,
        model: PreTrainedModel = None,
        split_name: str = None,
    ) -> Tuple[List[float], float]:

        predictions = Predictions(data={"filename": "", "values": generated_texts})
        diversity_metrics = {}
        msttr_metrics = self._msttr_metric.compute(None, predictions)
        n_gram_metrics = self._n_gram_metric.compute(None, predictions)

        for key, value in msttr_metrics.items():
            diversity_metrics[f"diversity_metrics/{key}"] = (None, value)
        for key, value in n_gram_metrics.items():
            diversity_metrics[f"diversity_metrics/{key}"] = (None, value)

        return diversity_metrics


class SummaCZSMetric(BaseMetric):
    """
    Consistency metric for summarization

    https://github.com/tingofurro/summac/
    """

    def __init__(self, **kwargs) -> None:
        super().__init__()
        self._scorer = SummaCZS(**kwargs)

    def compute(
        self,
        prompt_texts: List[str],
        generated_texts: List[str],
        reference_texts: List[List[str]],
        meta_infos: List[Dict[str, Any]] = None,
        model: PreTrainedModel = None,
        split_name: str = None,
    ) -> Tuple[List[float], float]:
        metric_results = self._scorer.score(prompt_texts, generated_texts)
        corpus_score = np.mean(metric_results["scores"])
        metric_dict = {"consistency/summaczs": (metric_results["scores"], corpus_score)}
        return metric_dict


class SummaCConvMetric(BaseMetric):
    """
    Consistency metric for summarization

    https://github.com/tingofurro/summac/
    """

    def __init__(self, **kwargs) -> None:
        super().__init__()
        self._scorer = SummaCConv(**kwargs)

    def compute(
        self,
        prompt_texts: List[str],
        generated_texts: List[str],
        reference_texts: List[List[str]],
        meta_infos: List[Dict[str, Any]] = None,
        model: PreTrainedModel = None,
        split_name: str = None,
    ) -> Tuple[List[float], float]:
        metric_results = self._scorer.score(prompt_texts, generated_texts)
        corpus_score = np.mean(metric_results["scores"])
        metric_dict = {
            "consistency/summacconv": (metric_results["scores"], corpus_score)
        }
        return metric_dict

class TargetQualityMetric(BaseMetric):
    """
    Metric to grade the quality of responses with respect to a trained target model.

    - This is a reference-free metric. So we will generate the score for both the model prediction and the ground-truth reference.
    - The metric essentially computes S(t) = M(t) / m
    
    M(k) refers to the predicted total points by the trained target model, given the utterances from 1 to k. M(k) only takes in the dialogue histories that end in the YOU utterance.

    m is the maximum possible score for that player (this formulation should work for both dealornodeal and CaSiNo).

    S(k) would then capture the fractional points expected by the model if the given utterance is played (or used as the next response).
    
    """
    def __init__(
        self,
        tokenizer_id: str,
        target_model_dir: str,
        padding_side: str = "right",
        truncation_side: str = "right",
        pad_token_as_eos_token: bool = False,
        max_length: int = 512,
        do_sample:bool = True,
        top_k:int = 50,
        top_p:float = 0.6,
        min_length:int = 5,
        num_beams:int = 1,
        max_new_tokens:int = 100,
    ) -> None:
        super().__init__()
        
        #set up the tokenizer
        self._tokenizer = AutoTokenizer.from_pretrained(tokenizer_id)
        self._tokenizer.truncation_side = truncation_side
        self._tokenizer.padding_side = padding_side
        if self._tokenizer.pad_token is None and pad_token_as_eos_token:
            self._tokenizer.pad_token = self._tokenizer.eos_token

        self._max_length = max_length
        self._batch_size = 16

        self._device = f"cuda:{torch.cuda.device_count() - 1}" if torch.cuda.is_available() else "cpu"

        # set up the model
        self._model = AutoModelForSeq2SeqLM.from_pretrained(target_model_dir).to(self._device)

        # set up generation params
        self._do_sample = do_sample
        self._top_k = top_k
        self._top_p = top_p
        self._min_length = min_length
        self._num_beams = num_beams
        self._max_new_tokens = max_new_tokens

    def extract_model_points(self, prompt_txts, out_txts, max_points):
        """Extract total points from generated texts. Return an avg score if the total could not be obtained. - half of max points may be."""
        
        model_points = []

        for prompt_txt, out_txt, max_point in zip(prompt_txts, out_txts, max_points):
            if not TargetFormatAndMetrics.has_good_form(prompt_txt, out_txt):
                model_points.append(max_point / 2)
                continue
            
            items = out_txt.split()
            f_ix = -1
            for ix, item in enumerate(items):
                if item == "food:":
                    f_ix = ix
                    break
            
            if f_ix == -1:
                return False
            
            model_points.append(int(items[f_ix + 20]))
        
        return model_points

    def extract_max_points(self, prompt_txts):
        """Extract maximum possible points from the prompt texts."""
        
        max_points = []
        for prompt in prompt_txts:
            # get pref scores
            prompt_items = prompt.split()
            prompt_f_ix = -1
            for ix, item in enumerate(prompt_items):
                if item == "food:":
                    prompt_f_ix = ix
                    break
            
            assert prompt_f_ix != -1

            f_c, w_c, fi_c = int(prompt_items[prompt_f_ix + 1][:-1]), int(prompt_items[prompt_f_ix + 4][:-1]), int(prompt_items[prompt_f_ix + 7][:-1])

            f_p, w_p, fi_p = int(prompt_items[prompt_f_ix + 2]), int(prompt_items[prompt_f_ix + 5]), int(prompt_items[prompt_f_ix + 8])

            mp = f_c*f_p + w_c*w_p + fi_c*fi_p
            max_points.append(mp)

        return max_points
    
    def extend_prompt(self, prompt_txt, new_txt):
        """Extend the prompt with the new txt so as to compute the score with the new utterance."""

        assert "<eou>" not in new_txt
        assert "<you>" in new_txt

        new_prompt_txt = prompt_txt.replace("<history>", f"<history> {new_txt}")

        return new_prompt_txt

    def compute(
        self,
        prompt_texts: List[str],
        generated_texts: List[str],
        reference_texts: List[List[str]],
        meta_infos: List[Dict[str, Any]] = None,
        model: PreTrainedModel = None,
        split_name: str = None,
    ) -> Tuple[List[float], float]:
        if split_name == "train":
            return {}

        pred_scores, ref_scores = [], []

        current_ix = 0
        n_texts = len(generated_texts)
        while current_ix < n_texts:
            batch_ref_texts = [ii[0] for ii in reference_texts[
                current_ix : current_ix + self._batch_size
            ]]
            batch_gen_texts = generated_texts[
                current_ix : current_ix + self._batch_size
            ]
            batch_prompt_texts = prompt_texts[
                current_ix : current_ix + self._batch_size
            ]

            gen_prompts = [self.extend_prompt(prompt_txt, gen_txt) for prompt_txt, gen_txt in zip(batch_prompt_texts, batch_gen_texts)]
            ref_prompts = [self.extend_prompt(prompt_txt, ref_txt) for prompt_txt, ref_txt in zip(batch_prompt_texts, batch_ref_texts)]

            gen_encodings = self._tokenizer(
                gen_prompts, return_tensors="pt", truncation=True, padding=True, max_length=self._max_length
            ).input_ids.to(self._device)
            ref_encodings = self._tokenizer(
                ref_prompts, return_tensors="pt", truncation=True, padding=True, max_length=self._max_length
            ).input_ids.to(self._device)

            with torch.no_grad():
                gen_outputs = model.generate(gen_encodings, do_sample=self._do_sample, top_k=self._top_k, top_p=self._top_p, min_length=self._min_length, num_beams=self._num_beams, max_new_tokens=self._max_new_tokens)

                ref_outputs = model.generate(ref_encodings, do_sample=self._do_sample, top_k=self._top_k, top_p=self._top_p, min_length=self._min_length, num_beams=self._num_beams, max_new_tokens=self._max_new_tokens)

            gen_dec = self._tokenizer.batch_decode(gen_outputs)
            ref_dec = self._tokenizer.batch_decode(ref_outputs)

            max_points = self.extract_max_points(batch_prompt_texts)

            gen_points = self.extract_model_points(batch_prompt_texts, gen_dec, max_points)
            ref_points = self.extract_model_points(batch_prompt_texts, ref_dec, max_points)

            assert type(max_points) == type(gen_points) == type(ref_points) == list
            assert len(max_points) == len(gen_points) == len(ref_points) == len(batch_ref_texts)

            pred_scores += [(gp / mp) for gp, mp in zip(gen_points, max_points)]
            ref_scores += [(rp / mp) for rp, mp in zip(ref_points, max_points)]

            current_ix += self._batch_size
        
        assert len(pred_scores) == len(ref_scores) == len(generated_texts)
        
        return {
            "target_metrics/gen_perf": (
                None,
                np.mean(pred_scores),
            ),
            "target_metrics/ref_perf": (
                None,
                np.mean(ref_scores),
            ),
            "target_metrics/total_count": (
                None,
                len(ref_scores),
            ),
        }

class Seq2SeqPerplexity(BaseMetric):
    def __init__(
        self,
        tokenizer_id: str,
        model_type: str = "seq2seq",
        padding_side: str = "right",
        truncation_side: str = "right",
        pad_token_as_eos_token: bool = False,
        max_length: int = 512,
    ) -> None:
        super().__init__()
        
        self._tokenizer = AutoTokenizer.from_pretrained(tokenizer_id)
        self._tokenizer.truncation_side = truncation_side
        self._tokenizer.padding_side = padding_side
        if self._tokenizer.pad_token is None and pad_token_as_eos_token:
            self._tokenizer.pad_token = self._tokenizer.eos_token

        self._max_length = max_length
        self._model_type = model_type
        self._batch_size = 16

    def get_device(self, model: PreTrainedModel):
        try:
            return model.transformer.first_device
        except:
            return model.device

    def compute(
        self,
        prompt_texts: List[str],
        generated_texts: List[str],
        reference_texts: List[List[str]],
        meta_infos: List[Dict[str, Any]] = None,
        model: PreTrainedModel = None,
        split_name: str = None,
    ) -> Tuple[List[float], float]:
        if split_name == "train":
            return {}

        if self._model_type != "seq2seq":
            raise NotImplementedError

        device = self.get_device(model)

        nlls = []
        current_ix = 0
        n_texts = len(generated_texts)
        while current_ix < n_texts:
            batch_ref_texts = [ii[0] for ii in reference_texts[
                current_ix : current_ix + self._batch_size
            ]]

            batch_prompt_texts = prompt_texts[
                current_ix : current_ix + self._batch_size
            ]

            encodings = self._tokenizer(
                batch_prompt_texts, return_tensors="pt", truncation=True, padding=True, max_length=self._max_length
            ).input_ids.to(device)

            ref_encodings = self._tokenizer(
                batch_ref_texts, return_tensors="pt", truncation=True, padding=True, max_length=self._max_length
            ).input_ids.to(device)

            ref_encodings[ref_encodings == 0] = -100

            with torch.no_grad():
                outputs = model(encodings, labels=ref_encodings)
                neg_log_likelihood = outputs[0]

            nlls.append(neg_log_likelihood)

            current_ix += self._batch_size
        
        return {
            "fluency_metrics/seq2seq_perplexity": (
                None,
                torch.exp(torch.mean(torch.stack(nlls))).item(),
            )
        }


class Perplexity(BaseMetric):
    def __init__(
        self,
        stride: int,
        tokenizer_id: str,
        model_type: str = "causal",
        use_text_from_meta_data: bool = False,
    ) -> None:
        super().__init__()
        self._tokenizer_id = tokenizer_id
        self._model_type = model_type
        self._stride = stride
        self._use_text_from_meta_data = use_text_from_meta_data

    def get_device(self, model: PreTrainedModel):
        try:
            return model.transformer.first_device
        except:
            return model.device

    def compute(
        self,
        prompt_texts: List[str],
        generated_texts: List[str],
        reference_texts: List[List[str]],
        meta_infos: List[Dict[str, Any]] = None,
        model: PreTrainedModel = None,
        split_name: str = None,
    ) -> Tuple[List[float], float]:
        if split_name == "train":
            return {}

        if self._model_type != "causal":
            raise NotImplementedError

        # we compute perplexity on reference texts
        if self._use_text_from_meta_data:
            reference_texts = [info["reference"] for info in meta_infos]
        else:
            reference_texts = [ref for refs in reference_texts for ref in refs]
        tokenizer = AutoTokenizer.from_pretrained(self._tokenizer_id)
        encodings = tokenizer("\n\n".join(reference_texts), return_tensors="pt")

        device = self.get_device(model)

        nlls = []
        max_length = model.config.n_positions
        for i in tqdm(range(0, encodings.input_ids.size(1), self._stride)):
            begin_loc = max(i + self._stride - max_length, 0)
            end_loc = min(i + self._stride, encodings.input_ids.size(1))
            trg_len = end_loc - i  # may be different from stride on last loop

            # run on last device
            input_ids = encodings.input_ids[:, begin_loc:end_loc].to(device)
            target_ids = input_ids.clone()
            target_ids[:, :-trg_len] = -100

            with torch.no_grad():
                outputs = model(input_ids, labels=target_ids)
                neg_log_likelihood = outputs[0] * trg_len

            nlls.append(neg_log_likelihood)

        return {
            "fluency_metrics/perplexity": (
                None,
                torch.exp(torch.stack(nlls).sum() / end_loc).item(),
            )
        }


class ParentToTTo:
    """
    Official version
    """

    def compute(
        self,
        prompt_texts: List[str],
        generated_texts: List[str],
        reference_texts: List[List[str]],
        meta_infos: List[Dict[str, Any]],
        model: PreTrainedModel = None,
        split_name: str = None,
    ):
        tables = [info["raw_table"] for info in meta_infos]
        parent_overall, parent_overlap, parent_non_overlap = compute_parent(
            generated_texts, tables
        )

        metric_results = {}
        metric_names = ["parent_overall", "parent_overlap", "parent_non_overlap"]
        metric_values = [parent_overall, parent_overlap, parent_non_overlap]
        for name, value in zip(metric_names, metric_values):
            metric_results[f"table_to_text/{name}/precision"] = (
                None,
                value["precision"],
            )
            metric_results[f"table_to_text/{name}/recall"] = (None, value["recall"])

            # individual f-scores - fetch only for overall since we don't know for which samples
            if name == "parent_overall":
                f_scores = value["all_f"]
            else:
                f_scores = None

            metric_results[f"table_to_text/{name}_f_score"] = (
                f_scores,
                value["f_score"],
            )
        return metric_results


class BLEUToTTo:
    """
    Official version
    """

    def compute(
        self,
        prompt_texts: List[str],
        generated_texts: List[str],
        reference_texts: List[List[str]],
        meta_infos: List[Dict[str, Any]],
        model: PreTrainedModel = None,
        split_name: str = None,
    ):
        tables = [info["raw_table"] for info in meta_infos]
        bleu_overall, bleu_overlap, bleu_non_overlap = compute_bleu(
            generated_texts, tables
        )

        metric_results = {
            "table_to_text/bleu_overall": (None, bleu_overall),
            "table_to_text/bleu_overlap": (None, bleu_overlap),
            "table_to_text/bleu_non_overlap": (None, bleu_non_overlap),
        }
        return metric_results


class RougeLMax(BaseMetric):
    def __init__(self, **args) -> None:
        super().__init__()
        self._metric = rouge.Rouge(metrics=["rouge-l"], **args)

    def _rouge_max_over_ground_truths(self, prediction, ground_truths):
        """
        Computes max of Rouge-L (https://github.com/allenai/unifiedqa/blob/bad6ef339db6286f0d8bd0661a2daeeb0f800f59/evaluation/evaluate_narrativeqa.py#L25)
        """
        # load stemmer
        self._metric.load_stemmer(self._metric.ensure_compatibility)

        scores_for_ground_truths = []
        for ground_truth in ground_truths:
            score = self._metric.get_scores(prediction, [ground_truth])
            scores_for_ground_truths.append(score)
        max_score = copy.deepcopy(score)
        max_score = max([score["rouge-l"]["f"] for score in scores_for_ground_truths])
        return max_score

    def compute(
        self,
        prompt_texts: List[str],
        generated_texts: List[str],
        reference_texts: List[List[str]],
        meta_infos: List[Dict[str, Any]] = None,
        model: PreTrainedModel = None,
        split_name: str = None,
    ):
        all_scores = []
        for gen_text, ref_texts in zip(generated_texts, reference_texts):
            rouge_max_score = self._rouge_max_over_ground_truths(gen_text, ref_texts)
            all_scores.append(rouge_max_score)

        metric_dict = {"lexical/rouge_l_max": (all_scores, np.mean(all_scores))}
        return metric_dict


class SacreBLEUMetric(BaseMetric):
    def __init__(self, **args) -> None:
        super().__init__()
        self._args = args
        self._metric = load_metric("sacrebleu")

    def compute(
        self,
        prompt_texts: List[str],
        generated_texts: List[str],
        reference_texts: List[List[str]],
        meta_infos: List[Dict[str, Any]] = None,
        model: PreTrainedModel = None,
        split_name: str = None,
    ) -> Tuple[List[float], float]:

        metric_results = self._metric.compute(
            predictions=generated_texts, references=reference_texts, **self._args
        )
        bleu_score = metric_results["score"] / 100
        metric_dict = {"lexical/sacrebleu": (None, bleu_score)}
        return metric_dict

class TargetFormatAndMetrics(BaseMetric):
    """
    Metric used for training the target model. This captures the RMSE and format checking to understand how the target model has been trained.
    """
    def __init__(self) -> None:
        super().__init__()

    @staticmethod
    def has_good_form(prompt, txt):
        """Check if the format of the generated text is good."""

        if "total points: " not in txt:
            return False
        
        if "food: " not in txt:
            return False
        
        if "water: " not in txt:
            return False
        
        if "firewood: " not in txt:
            return False
        
        items = txt.split()

        if len(items) < 21:
            return False

        f_ix = -1
        for ix, item in enumerate(items):
            if item == "food:":
                f_ix = ix
                break
        
        if f_ix == -1:
            return False
        
        try:
            # get deal counts
            f_c, w_c, fi_c = int(items[f_ix + 1][:-1]), int(items[f_ix + 3][:-1]), int(items[f_ix + 5][:-1])
        except:
            return False
        
        # get pref scores
        prompt_items = prompt.split()
        prompt_f_ix = -1
        for ix, item in enumerate(prompt_items):
            if item == "food:":
                prompt_f_ix = ix
                break
        
        assert prompt_f_ix != -1

        f_p, w_p, fi_p = int(prompt_items[prompt_f_ix + 2]), int(prompt_items[prompt_f_ix + 5]), int(prompt_items[prompt_f_ix + 8])

        if items[f_ix + 8] != f"{f_c}*{f_p}":
            return False

        if items[f_ix + 10] != f"{w_c}*{w_p}":
            return False

        if items[f_ix + 12] != f"{fi_c}*{fi_p}":
            return False

        try:
            if int(items[f_ix + 14]) != f_c*f_p:
                return False

            if int(items[f_ix + 16]) != w_c*w_p:
                return False

            if int(items[f_ix + 18]) != fi_c*fi_p:
                return False
            
            if int(items[f_ix + 20]) != (f_c*f_p + w_c*w_p + fi_c*fi_p):
                return False
        except:
            return False

        return True

    @staticmethod
    def get_rmse(pred, truth):
        """Compute rmse score. Technically, just compute the se over here. complete rmse will be computed above in the pipeline."""

        return (pred - truth)**2

    @staticmethod
    def get_acc(pred, truth, width):
        """Get the accuracy +- of the width."""

        allowed_min = truth - width
        allowed_max = truth + width

        if allowed_min <= pred <= allowed_max:
            return 1.0
        else:
            return 0.0

    def compute_metric_scores(self, pred, ref):
        """Compute metric values."""

        pred_items = pred.split()

        f_ix = -1
        for ix, item in enumerate(pred_items):
            if item == "food:":
                f_ix = ix
                break
        assert f_ix != -1

        #predictions
        f_p, w_p, fi_p, total_p = int(pred_items[f_ix + 1][:-1]), int(pred_items[f_ix + 3][:-1]), int(pred_items[f_ix + 5][:-1]), int(pred_items[f_ix + 20])

        ref_items = ref.split()

        f_ix = -1
        for ix, item in enumerate(ref_items):
            if "food:" in item:
                f_ix = ix
                break
        assert f_ix != -1

        #ground truth
        f_r, w_r, fi_r, total_r = int(ref_items[f_ix + 1][:-1]), int(ref_items[f_ix + 3][:-1]), int(ref_items[f_ix + 5][:-1]), int(ref_items[f_ix + 20])

        # priorities
        f_pref, w_pref, fi_pref = int(ref_items[f_ix + 8].split("*")[-1]), int(ref_items[f_ix + 10].split("*")[-1]), int(ref_items[f_ix + 12].split("*")[-1])

        # random predictions
        f_rand, w_rand, fi_rand = random.choice([0,1,2,3]), random.choice([0,1,2,3]), random.choice([0,1,2,3])
        total_rand = f_rand*f_pref + w_rand*w_pref + fi_rand*fi_pref

        this_metric_dict = {
            "rmse_food": TargetFormatAndMetrics.get_rmse(pred=f_p, truth=f_r),
            "rmse_water": TargetFormatAndMetrics.get_rmse(pred=w_p, truth=w_r),
            "rmse_firewood": TargetFormatAndMetrics.get_rmse(pred=fi_p, truth=fi_r),
            "rmse_total_points": TargetFormatAndMetrics.get_rmse(pred=total_p, truth=total_r),

            "acc_0_food": TargetFormatAndMetrics.get_acc(pred=f_p, truth=f_r, width=0),
            "acc_0_water": TargetFormatAndMetrics.get_acc(pred=w_p, truth=w_r, width=0),
            "acc_0_firewood": TargetFormatAndMetrics.get_acc(pred=fi_p, truth=fi_r, width=0),
            "acc_0_total": TargetFormatAndMetrics.get_acc(pred=total_p, truth=total_r, width=0),
            "acc_1_food": TargetFormatAndMetrics.get_acc(pred=f_p, truth=f_r, width=1),
            "acc_1_water": TargetFormatAndMetrics.get_acc(pred=w_p, truth=w_r, width=1),
            "acc_1_firewood": TargetFormatAndMetrics.get_acc(pred=fi_p, truth=fi_r, width=1),
            "acc_1_total": TargetFormatAndMetrics.get_acc(pred=total_p, truth=total_r, width=1),
            "acc_2_food": TargetFormatAndMetrics.get_acc(pred=f_p, truth=f_r, width=2),
            "acc_2_water": TargetFormatAndMetrics.get_acc(pred=w_p, truth=w_r, width=2),
            "acc_2_firewood": TargetFormatAndMetrics.get_acc(pred=fi_p, truth=fi_r, width=2),
            "acc_2_total": TargetFormatAndMetrics.get_acc(pred=total_p, truth=total_r, width=2),

            "rmse_food_rand": TargetFormatAndMetrics.get_rmse(pred=f_rand, truth=f_r),
            "rmse_water_rand": TargetFormatAndMetrics.get_rmse(pred=w_rand, truth=w_r),
            "rmse_firewood_rand": TargetFormatAndMetrics.get_rmse(pred=fi_rand, truth=fi_r),
            "rmse_points_rand": TargetFormatAndMetrics.get_rmse(pred=total_rand, truth=total_r),

            "acc_0_food_rand": TargetFormatAndMetrics.get_acc(pred=f_rand, truth=f_r, width=0),
            "acc_0_water_rand": TargetFormatAndMetrics.get_acc(pred=w_rand, truth=w_r, width=0),
            "acc_0_firewood_rand": TargetFormatAndMetrics.get_acc(pred=fi_rand, truth=fi_r, width=0),
            "acc_0_total_rand": TargetFormatAndMetrics.get_acc(pred=total_rand, truth=total_r, width=0),
            "acc_1_food_rand": TargetFormatAndMetrics.get_acc(pred=f_rand, truth=f_r, width=1),
            "acc_1_water_rand": TargetFormatAndMetrics.get_acc(pred=w_rand, truth=w_r, width=1),
            "acc_1_firewood_rand": TargetFormatAndMetrics.get_acc(pred=fi_rand, truth=fi_r, width=1),
            "acc_1_total_rand": TargetFormatAndMetrics.get_acc(pred=total_rand, truth=total_r, width=1),
            "acc_2_food_rand": TargetFormatAndMetrics.get_acc(pred=f_rand, truth=f_r, width=2),
            "acc_2_water_rand": TargetFormatAndMetrics.get_acc(pred=w_rand, truth=w_r, width=2),
            "acc_2_firewood_rand": TargetFormatAndMetrics.get_acc(pred=fi_rand, truth=fi_r, width=2),
            "acc_2_total_rand": TargetFormatAndMetrics.get_acc(pred=total_rand, truth=total_r, width=2),
        }

        return this_metric_dict

    def compute(
        self,
        prompt_texts: List[str],
        generated_texts: List[str],
        reference_texts: List[List[str]],
        meta_infos: List[Dict[str, Any]] = None,
        model: PreTrainedModel = None,
        split_name: str = None,
    ) -> Tuple[List[float], float]:

        good_forms = []
        metric_dict = {
            "rmse_food": 0.0,
            "rmse_water": 0.0,
            "rmse_firewood": 0.0,
            "rmse_total_points": 0.0,
            "acc_0_food": 0.0,
            "acc_0_water": 0.0,
            "acc_0_firewood": 0.0,
            "acc_0_total": 0.0,
            "acc_1_food": 0.0,
            "acc_1_water": 0.0,
            "acc_1_firewood": 0.0,
            "acc_1_total": 0.0,
            "acc_2_food": 0.0,
            "acc_2_water": 0.0,
            "acc_2_firewood": 0.0,
            "acc_2_total": 0.0,
            "rmse_food_rand": 0.0,
            "rmse_water_rand": 0.0,
            "rmse_firewood_rand": 0.0,
            "rmse_points_rand": 0.0,
            "acc_0_food_rand": 0.0,
            "acc_0_water_rand": 0.0,
            "acc_0_firewood_rand": 0.0,
            "acc_0_total_rand": 0.0,
            "acc_1_food_rand": 0.0,
            "acc_1_water_rand": 0.0,
            "acc_1_firewood_rand": 0.0,
            "acc_1_total_rand": 0.0,
            "acc_2_food_rand": 0.0,
            "acc_2_water_rand": 0.0,
            "acc_2_firewood_rand": 0.0,
            "acc_2_total_rand": 0.0,

        }

        for prompt, pred, refs in zip(prompt_texts, generated_texts, reference_texts):

            if not TargetFormatAndMetrics.has_good_form(prompt, pred):
                good_forms.append(0)
                continue

            good_forms.append(1)

            ref = refs[0]

            this_metric_dict = self.compute_metric_scores(pred, ref)

            for k in this_metric_dict.keys():
                metric_dict[k] += this_metric_dict[k]
        
        good_form_acc = sum(good_forms) / len(generated_texts)

        out_dict = {
            "nego_target/format_accuracy": (good_forms, good_form_acc),
            "nego_target/total_count": (None, len(generated_texts)),
            "nego_target/good_count": (None, sum(good_forms)),
            }

        for k in metric_dict.keys():
            if sum(good_forms) > 0:

                if "rmse" in k:
                    metric_dict[k] = (metric_dict[k] / sum(good_forms))**0.5
                    out_dict[f"nego_target/{k}"] = (None, metric_dict[k])
                elif "acc_" in k:
                    metric_dict[k] = metric_dict[k] / sum(good_forms)
                    out_dict[f"nego_target/{k}"] = (None, metric_dict[k])
                else:
                    raise ValueError
            else:
                out_dict[f"nego_target/{k}"] = (None, -1)

        return out_dict

class TERMetric(BaseMetric):
    def __init__(self) -> None:
        super().__init__()
        self._metric = load_metric("ter")

    def compute(
        self,
        prompt_texts: List[str],
        generated_texts: List[str],
        reference_texts: List[List[str]],
        meta_infos: List[Dict[str, Any]] = None,
        model: PreTrainedModel = None,
        split_name: str = None,
    ) -> Tuple[List[float], float]:

        metric_results = self._metric.compute(
            predictions=generated_texts, references=reference_texts
        )
        score = metric_results["score"] / 100
        metric_dict = {"lexical/ter": (None, score)}
        return metric_dict


class chrFmetric(BaseMetric):
    def __init__(self) -> None:
        super().__init__()
        self._metric = load_metric("chrf")

    def compute(
        self,
        prompt_texts: List[str],
        generated_texts: List[str],
        reference_texts: List[List[str]],
        meta_infos: List[Dict[str, Any]] = None,
        model: PreTrainedModel = None,
        split_name: str = None,
    ) -> Tuple[List[float], float]:

        metric_results = self._metric.compute(
            predictions=generated_texts, references=reference_texts
        )
        score = metric_results["score"] / 100
        metric_dict = {"lexical/chrf": (None, score)}
        return metric_dict


class IntentAccuracyDailyDialog(BaseMetric):
    def __init__(self) -> None:
        super().__init__()
        self._tokenizer = AutoTokenizer.from_pretrained(
            "rajkumarrrk/roberta-daily-dialog-intent-classifier"
        )
        self._model = AutoModelForSequenceClassification.from_pretrained(
            "rajkumarrrk/roberta-daily-dialog-intent-classifier"
        )
        self._device = "cuda" if torch.cuda.is_available() else "cpu"
        self._device = f"cuda:{torch.cuda.device_count() - 1}"
        self._model = self._model.to(self._device)

    def compute(
        self,
        prompt_texts: List[str],
        generated_texts: List[str],
        reference_texts: List[List[str]],
        meta_infos: List[Dict[str, Any]] = None,
        model: PreTrainedModel = None,
        split_name: str = None,
    ) -> Tuple[List[float], float]:
        def get_input_for_classifier(prompt, generated_text):
            history = prompt.split(DailyDialog.EOU_TOKEN)
            history = [utt for utt in history if utt != ""]
            last_utterance = history[-1]
            input_text = last_utterance + generated_text
            return input_text

        # we have to extract the history utterances
        input_texts = [
            get_input_for_classifier(prompt, gen)
            for prompt, gen in zip(prompt_texts, generated_texts)
        ]

        # extract target intents
        target_intents = [info["intent"][0] - 1 for info in meta_infos]

        # tokenize
        encoded = self._tokenizer(
            input_texts, return_tensors="pt", truncation=True, padding=True
        )

        with torch.no_grad():
            outputs = self._model(
                input_ids=encoded.input_ids.to(self._device),
                attention_mask=encoded.attention_mask.to(self._device),
            )
            pred_labels = torch.argmax(outputs.logits, dim=1).tolist()

        matching_scores = (np.array(pred_labels) == np.array(target_intents)).astype(
            np.int32
        )
        intent_accuracy = np.mean(matching_scores)

        metric_dict = {"intent/accuracy": (matching_scores.tolist(), intent_accuracy)}
        return metric_dict


if __name__ == "__main__":
    prompt_texts = [""]
    gen_texts = ["Hello there general kenobi", "foo bar foobar"]
    reference_texts = [["Hello there general kenobi"], ["foo bar foobar"]]
    # metric = MeteorMetric()
    # print(metric.compute(prompt_texts, gen_texts, reference_texts))

    # metric = RougeMetric()
    # print(metric.compute(prompt_texts, gen_texts, reference_texts))

    # metric = SacreBLEUMetric(tokenize="intl")
    # print(metric.compute(prompt_texts, gen_texts, reference_texts))

    # metric = TERMetric()
    # print(metric.compute(prompt_texts, gen_texts, reference_texts))

    # metric = chrFmetric()
    # print(metric.compute(prompt_texts, gen_texts, reference_texts))

    # metric = BERTScoreMetric(language="en")
    # print(metric.compute(prompt_texts, gen_texts, reference_texts))

    # metric = BLEUMetric()
    # print(metric.compute(prompt_texts, gen_texts, reference_texts))

    # metric = BLEURTMetric()
    # print(metric.compute(prompt_texts, gen_texts, reference_texts))

    # metric = DiversityMetrics()
    # print(metric.compute(prompt_texts, gen_texts, reference_texts))

    # document = """Jeff joined Microsoft in 1992 to lead corporate developer evangelism for Windows NT. He then served as a Group Program manager in Microsoft’s Internet Business Unit. In 1998, he led the creation of SharePoint Portal Server, which became one of Microsoft’s fastest-growing businesses, exceeding $2 billion in revenues. Jeff next served as Corporate Vice President for Program Management across Office 365 Services and Servers, which is the foundation of Microsoft’s enterprise cloud leadership. He then led Corporate Strategy supporting Satya Nadella and Amy Hood on Microsoft’s mobile-first/cloud-first transformation and acquisitions. Prior to joining Microsoft, Jeff was vice president for software development for an investment firm in New York. He leads Office shared experiences and core applications, as well as OneDrive and SharePoint consumer and business services in Office 365. Jeff holds a Master of Business Administration degree from Harvard Business School and a Bachelor of Science degree in information systems and finance from New York University."""
    # summary = "Jeff joined Microsoft in 1992 to lead the company's corporate evangelism. He then served as a Group Manager in Microsoft's Internet Business Unit. In 1998, Jeff led Sharepoint Portal Server, which became the company's fastest-growing business, surpassing $3 million in revenue. Jeff next leads corporate strategy for SharePoint and Servers which is the basis of Microsoft's cloud-first strategy. He leads corporate strategy for Satya Nadella and Amy Hood on Microsoft's mobile-first."

    # metric = SummaCZSMetric(granularity="sentence",
    #                         use_ent=True,
    #                         use_con=False)
    # print(metric.compute([document], [summary], []))

    # metric = SummaCConvMetric(granularity="sentence")
    # print(metric.compute([document], [summary], []))

    prompt_texts = ["1", "2"]
    gen_texts = [
        "The dog is the boy's cat.",
        "A boy is picking apples from trees and put them into bags.",
    ]
    reference_texts = [
        ["The dog is the boy's cat.", "The dog eats the cat of the boy."],
        ["A boy is picking apples from trees."],
    ]
    metric = CIDERMetric()
    print(metric.compute(prompt_texts, gen_texts, reference_texts))

    metric = SpiceMetric()
    print(metric.compute(prompt_texts, gen_texts, reference_texts))
