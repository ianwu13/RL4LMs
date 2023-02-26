import os
from bot_bot_interaction.predict_agreed_deal import PredictAgreedDeal
import torch # type: ignore
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM # type: ignore
from torch.nn import CrossEntropyLoss # type: ignore

class Agent:
    def __init__(self, config, name) -> None:
        """Load the model, tokenizer, define any internal states."""
        self.config = config
        self.name = name

    def reset_agent(self, cxt):
        """Reset the agent state at the start of a new conversation and set up the new agent context."""
        raise NotImplementedError

    def receive(self, resp_obj):
        """Provided resp_obj came from the opponent. Update the internal states based on the received resp."""
        raise NotImplementedError
    
    def respond(self):
        """Respond, update your internal state, and send out the resp."""
        raise NotImplementedError

class SupervisedAgent(Agent):
    def __init__(self, config, name) -> None:
        super().__init__(config, name)

        tokenizer_id = "t5-base"
        padding_side = "right"
        truncation_side = "right"
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_id)
        self.tokenizer.truncation_side = truncation_side
        self.tokenizer.padding_side = padding_side

        model_path = os.path.join(self.config.log_dir, self.name.split("_typ_")[0], "model")
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_path)

        # device
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = self.model.to(self.device)

        self.input_max_length = 512
        self.max_new_tokens = 64
        self.num_beams = 1
        self.min_length = 3
        self.do_sample = True
        self.top_p = 0.6
        self.top_k = 100

        print(f"Agent initialized: {self.name}")

    def reset_agent(self, cxt):
        """Reset the agent."""
        self.cxt = cxt
        self.input_seq = f"<context> {cxt} <history>"
    
    def receive(self, resp_obj):
        """Receive a response object."""
        assert resp_obj["name"] != self.name # came from the opponent

        # add to the start of the input_seq.
        self.input_seq = self.input_seq.replace("<history>", f"<history> <them> {resp_obj['resp']}")

    def respond(self):
        """Respond to the input request."""
        
        input_ids = self.tokenizer(self.input_seq, return_tensors="pt", truncation=True, padding=True, max_length=self.input_max_length).input_ids
        out_encs = self.model.generate(
            input_ids.to(self.device),
            num_beams=self.num_beams,
            do_sample=self.do_sample, 
            top_p=self.top_p,
            top_k=self.top_k,
            max_new_tokens=self.max_new_tokens,
            min_length=self.min_length
            )
        resp = self.tokenizer.decode(out_encs[0].to("cpu"),
                                        max_length=self.max_new_tokens,
                                        truncation=True)
        resp = resp.replace("<unk>you>", "").replace("<unk>EOU>", "").strip()
        resp = resp.replace("<pad>", "").replace("</s>", "").replace("<unk>","").strip()
        resp = resp.replace("selection>", "<selection>").strip()

        # add response to the start of the history.
        self.input_seq = self.input_seq.replace("<history>", f"<history> <you> {resp}")

        resp_obj = {
            "resp": resp,
            "name": self.name,
        }

        return resp_obj

"""
Agent that uses supervised learning + ranks candidates based on a score.
"""
class SupervisedRankingAgent(Agent):
    def __init__(self, config, name) -> None:
        super().__init__(config, name)

        tokenizer_id = "t5-base"
        padding_side = "right"
        truncation_side = "right"
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_id)
        self.tokenizer.truncation_side = truncation_side
        self.tokenizer.padding_side = padding_side

        model_path = os.path.join(self.config.log_dir, self.name.split("_typ_")[0], "model")
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_path)

        # to be used for scoring the generated candidates.
        self.predict_deal_obj = PredictAgreedDeal(self.config)

        # device
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = self.model.to(self.device)

        self.input_max_length = 512
        self.max_new_tokens = 64
        self.num_beams = 1
        self.min_length = 3
        self.do_sample = True
        self.top_p = 0.8
        self.top_k = 100
        self.num_return_sequences = 8

        print(f"Agent initialized: {self.name}")

    def reset_agent(self, cxt):
        """Reset the agent."""
        self.cxt = cxt
        self.input_seq = f"<context> {cxt} <history>"
    
    def receive(self, resp_obj):
        """Receive a response object."""
        assert resp_obj["name"] != self.name # came from the opponent

        # add to the start of the input_seq.
        self.input_seq = self.input_seq.replace("<history>", f"<history> <them> {resp_obj['resp']}")

    def respond(self):
        """Respond to the input request.
        Generate k samples. and then rank them according to a score.
        """
        
        input_ids = self.tokenizer(self.input_seq, return_tensors="pt", truncation=True, padding=True, max_length=self.input_max_length).input_ids
        out_encs = self.model.generate(
            input_ids.to(self.device),
            num_beams=self.num_beams,
            do_sample=self.do_sample, 
            top_p=self.top_p,
            top_k=self.top_k,
            max_new_tokens=self.max_new_tokens,
            min_length=self.min_length,
            num_return_sequences=self.num_return_sequences,
            )

        cands = []
        for ix in range(self.num_return_sequences):
            cand = self.tokenizer.decode(out_encs[ix].to("cpu"),
                                            max_length=self.max_new_tokens,
                                            truncation=True)
            cand = cand.replace("<unk>you>", "").replace("<unk>EOU>", "").strip()
            cand = cand.replace("<pad>", "").replace("</s>", "").replace("<unk>","").strip()
            cand = cand.replace("selection>", "<selection>").strip()
            
            cands.append(cand)

        cand2scores = self.compute_cand_scores(cands)

        resp = self.select_best_cand(cand2scores)

        # add response to the start of the history.
        self.input_seq = self.input_seq.replace("<history>", f"<history> <you> {resp}")

        resp_obj = {
            "resp": resp,
            "name": self.name,
        }

        return resp_obj

    def get_prompt_texts(self, cands):
        """Get the prompt texts."""
        prompt_texts = [self.input_seq for _ in range(len(cands))]
        return prompt_texts

    def get_pred_texts(self, cands):
        """Get the pred texts."""

        pred_texts = []
        for cand in cands:
            pred = f"<you> {cand} <EOU>"
            pred_texts.append(pred)
        return pred_texts

    def get_gen_scores(self, cands):
        """Compute the gen probability for each candidate, as per the base
        response generation model. - perform it batchwise."""
        
        batch_prompt_texts = self.get_prompt_texts(cands)
        batch_pred_texts = self.get_pred_texts(cands)

        encodings = self.tokenizer(
                    batch_prompt_texts, return_tensors="pt", truncation=True, padding=True, max_length=self.input_max_length
                )

        pred_encodings = self.tokenizer(
                    batch_pred_texts, return_tensors="pt", truncation=True, padding=True, max_length=self.max_new_tokens
                ).input_ids

        pred_encodings[pred_encodings == 0] = -100

        with torch.no_grad():
            outputs = self.model(
                encodings.input_ids.to(self.device),
                attention_mask=encodings.attention_mask.to(self.device),
                labels=pred_encodings.to(self.device))

        logits = outputs[1]

        loss_fct = CrossEntropyLoss()

        scores = []
        for ix in range(len(cands)):
            loss = loss_fct(logits[ix].reshape(-1, self.model.decoder.config.vocab_size), pred_encodings[ix].to(self.device).view(-1))
            scores.append(-loss.to("cpu").item())
            
        cand2scores = {}
        for cand, score in zip(cands, scores):
            cand2scores[cand] = score
        return cand2scores

    def compute_cand_scores(self, cands):
        """Compute various kinds of scores for the candidates.
        One can be perplexity score based on the generation model,
        - One can be some combination of the points that will be scored if the conv
        ends here and the confidence of the final prediction model.
        """

        cand2scores = {}
        for cand in cands:
            cand2scores[cand] = {}
        
        #dict cand -> prob
        gen_scores = self.get_gen_scores(cands)

        # dict cand -> deal + prob.
        deal_dists = self.predict_deal_obj.get_deal_dists(self.input_seq, cands)

        for cand in cands:
            cand2scores[cand]["gen_score"] = gen_scores[cand]
            cand2scores[cand]["deal_dist"] = deal_dists[cand]

        return cand2scores

    def get_pref_values(self, cxt):
        """Get issue-wise preference values from the given cxt."""
        items = cxt.strip().split()
        assert len(items) == 11

        i1_c, i2_c, i3_c = int(items[3]), int(items[6].rstrip(",")), int(items[9])
        pps = [i1_c, i2_c, i3_c]

        issues = None
        if "book" in cxt:
            issues = ["book", "hat", "ball"]
        else:
            issues = ["food", "water", "firewood"]
        
        prefs = {}

        for iss, pp in zip(issues, pps):
            prefs[iss] = pp

        return prefs

    def get_you_points(self, deal):
        """Get points of you player based on the given deal."""
        
        # extract the context from the input_seq
        cxt = self.input_seq.split("<history>")[0].replace("<context>","").strip()

        # now get the preferences of the you player from the cxt.
        prefs = self.get_pref_values(cxt)

        # now get the you counts from the deal
        deal_items = deal.split("<them>")[0].strip().split()
        deal_counts = {}
        for item in deal_items:
            if "=" in item:
                iis = item.split("=")
                deal_counts[iis[0]] = int(iis[1])
        assert len(deal_counts) == 3

        points = 0
        for issue, p in prefs.items():
            points += deal_counts[issue]*p
        
        return points

    def compute_cand_quality(self, scores):
        """Compute the final quality score of a candidate based on the available scores."""

        # compute the expected points scored.
        ep = 0.0
        for row in scores["deal_dist"]:
            deal, value = row[0], row[1]
            ep += self.get_you_points(deal)*value
        return ep

    def select_best_cand(self, cand2scores):
        """Select the best candidate.
        
        How to rank:
        - just use the expected points that can be scored by choosing this deal.
        """

        max_cand, max_q = None, float("-inf")

        for cand, scores in cand2scores.items():
            this_quality = self.compute_cand_quality(scores)
            if this_quality > max_q:
                max_q = this_quality
                max_cand = cand
        assert max_cand
        return max_cand

class OfflineRLAgent(Agent):
    def __init__(self, config, name) -> None:
        super().__init__(config, name)

        self.initial_rtg = 0

        tokenizer_id = "t5-base"
        padding_side = "right"
        truncation_side = "right"
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_id)
        self.tokenizer.truncation_side = truncation_side
        self.tokenizer.padding_side = padding_side

        model_path = os.path.join(self.config.log_dir, self.name.split("_typ_")[0], "model")
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_path)

        # device
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = self.model.to(self.device)

        self.input_max_length = 512
        self.max_new_tokens = 64
        self.num_beams = 1
        self.min_length = 3
        self.do_sample = True
        self.top_p = 0.6
        self.top_k = 100

        print(f"Agent initialized: {self.name}")

    def reset_agent(self, cxt):
        """Reset the agent."""
        self.cxt = cxt
        self.input_seq = f"<rewards> {self.initial_rtg} <context> {cxt} <history>"

    def get_new_rtg(self):
        """Get the new rtg."""
        prev_rtg = self.input_seq.split("<context>")[0].strip()
        prev_rtg = prev_rtg.split("<rewards>")[-1].strip()
        prev_rtg = int(prev_rtg.split()[-1])

        # just the reward for a new utterance for now.
        score = -2
        new_rtg = prev_rtg - score

        return new_rtg
    
    def receive(self, resp_obj):
        """Receive a response object."""
        assert resp_obj["name"] != self.name # came from the opponent

        # add to the start of the input_seq.
        self.input_seq = self.input_seq.replace("<history>", f"<history> <them> {resp_obj['resp']}")

    def respond(self):
        """Respond to the input request."""
        
        input_ids = self.tokenizer(self.input_seq, return_tensors="pt", truncation=True, padding=True, max_length=self.input_max_length).input_ids
        out_encs = self.model.generate(
            input_ids.to(self.device),
            num_beams=self.num_beams,
            do_sample=self.do_sample, 
            top_p=self.top_p,
            top_k=self.top_k,
            max_new_tokens=self.max_new_tokens,
            min_length=self.min_length
            )
        resp = self.tokenizer.decode(out_encs[0].to("cpu"),
                                        max_length=self.max_new_tokens,
                                        truncation=True)
        resp = resp.replace("<unk>you>", "").replace("<unk>EOU>", "").strip()
        resp = resp.replace("<pad>", "").replace("</s>", "").replace("<unk>","").strip()
        resp = resp.replace("selection>", "<selection>").strip()

        # add response to the start of the history.
        self.input_seq = self.input_seq.replace("<history>", f"<history> <you> {resp}")

        #also add the new rtg
        new_rtg = self.get_new_rtg()
        self.input_seq = self.input_seq.replace("<context>", f"{new_rtg} <context>")

        resp_obj = {
            "resp": resp,
            "name": self.name,
        }

        return resp_obj