"""
Class to use the trained model for predicting the agreed deal from a finished conversation.
"""

import operator
import numpy as np
import torch # type: ignore
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM # type: ignore
from torch.nn import CrossEntropyLoss # type: ignore

class PredictAgreedDeal:
    def __init__(self, config) -> None:
        """Load the model, tokenizer, define any internal states."""
        self.config = config

        tokenizer_id = "t5-base"
        padding_side = "right"
        truncation_side = "right"
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_id)
        self.tokenizer.truncation_side = truncation_side
        self.tokenizer.padding_side = padding_side

        self.model = AutoModelForSeq2SeqLM.from_pretrained(self.config.predict_deal_model_path)

        # device
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = self.model.to(self.device)

        self.input_max_length = 512
        self.max_new_tokens = 64
        self.num_beams = 1
        self.min_length = 3
        self.do_sample = False

        print(f"Predict deal model initialized.")

    def get_count_str(self, conv):
        """
        get the count string.
        """

        cxt = list(conv["cxts"].values())[0]
        items = cxt.strip().split()
        assert len(items) == 11

        i1_c, i2_c, i3_c = int(items[2].rstrip(",")), int(items[5].rstrip(",")), int(items[8].rstrip(","))
        
        if "book" in cxt:
            count_str = f"<counts> book={i1_c} hat={i2_c} ball={i3_c}"
        else:
            assert "food" in cxt
            count_str = f"<counts> food={i1_c} water={i2_c} firewood={i3_c}"

        return count_str
        

    def get_dialogue(self, mod2ano, conv):
        """Get the dialogue list along with speaker id."""
        dial = []

        for utt in conv["utts"]:
            this_item = f"{mod2ano[utt['name']]} {utt['resp']}"
            dial.append(this_item)

        return dial

    def get_input_seq(self, count_str, dialogue):
        """Construct the input sequence."""
        dial2 = dialogue[:]
        dial2.reverse()

        dial2 = " ".join(dial2)

        input_seq = f"{count_str} <history> {dial2}".strip()

        return input_seq

    def format_resp(self, resp):
        """Format the response."""

        resp = resp.replace("<pad>", "").replace("</s>", "").strip()
        resp = resp.replace("<unk>","").replace("EOU>", "").strip()
        resp = resp.replace("alice>", "<alice>").replace("bob>", "<bob>").replace("<EOU>","").strip()
        
        return resp

    def extract_deal(self, pred, ano2mod):
        """Extract the deal from the model prediction."""

        deal = {}
        alice_stuff, bob_stuff = pred.split("<bob>")[0].strip().split(), pred.split("<bob>")[-1].strip().split()

        # alice_stuff
        deal[ano2mod["<alice>"]] = {}
        for item in alice_stuff:
            if "=" in item:
                ii = item.split("=")
                deal[ano2mod["<alice>"]][ii[0]] = int(ii[1])

        # bob_stuff
        deal[ano2mod["<bob>"]] = {}
        for item in bob_stuff:
            if "=" in item:
                ii = item.split("=")
                deal[ano2mod["<bob>"]][ii[0]] = int(ii[1])
        
        return deal

    def has_good_form(self, conv, txt):
        """Check if the format of the generated text is good."""

        if "<alice>" not in txt or "<bob>" not in txt:
            return False

        if txt.count("=") != 6:
            return False
        
        if "book" not in txt and "food" not in txt:
            return False
        elif "book" in txt:
            if "food" in txt:
                return False
            if "ball" not in txt and "hat" not in txt:
                return False
            
            if txt.count("book=") != 2:
                return False
        
            if txt.count("hat=") != 2:
                return False

            if txt.count("ball=") != 2:
                return False
        elif "food" in txt:
            if "ball" in txt:
                return False
            if "water" not in txt and "firewood" not in txt:
                return False
            if txt.count("food=") != 2:
                return False
        
            if txt.count("water=") != 2:
                return False

            if txt.count("firewood=") != 2:
                return False

        # counts of the three items.

        cxt = list(conv["cxts"].values())[0]
        items = cxt.strip().split()
        assert len(items) == 11

        i1_c, i2_c, i3_c = int(items[2].rstrip(",")), int(items[5].rstrip(",")), int(items[8].rstrip(","))
        cnts = [i1_c, i2_c, i3_c]

        try:
            # all 6 values
            txt_items = txt.split()

            deal_cnts = []
            for item in txt_items:
                if "=" in item:
                    deal_cnts.append(int(item.split("=")[-1]))

            if len(deal_cnts) != 6:
                return False
            
            if (deal_cnts[0] + deal_cnts[3]) != cnts[0]:
                return False

            if (deal_cnts[1] + deal_cnts[4]) != cnts[1]:
                return False
            
            if (deal_cnts[2] + deal_cnts[5]) != cnts[2]:
                return False
        except:
            return False

        return True
    
    def get_deal(self, conv):
        """Infer from the trained model and return the deal that the players agreed in the given conv."""
        
        mod2ano, ano2mod = {}, {}
        ano_names = ["<alice>", "<bob>"]
        for mname, ano_name in zip(conv["cxts"].keys(), ano_names):
            mod2ano[mname] = ano_name
            ano2mod[ano_name] = mname
        
        count_str = self.get_count_str(conv)
        
        dialogue = self.get_dialogue(mod2ano, conv)

        input_seq = self.get_input_seq(count_str, dialogue)

        input_ids = self.tokenizer(input_seq, return_tensors="pt", truncation=True, padding=True, max_length=self.input_max_length).input_ids

        out_encs = self.model.generate(input_ids.to(self.device),
                    num_beams=self.num_beams,
                    do_sample=self.do_sample, 
                    max_new_tokens=self.max_new_tokens,
                    min_length=self.min_length
                    )

        pred = self.format_resp(self.tokenizer.decode(out_encs[0].to("cpu"),
                                    max_length=self.max_new_tokens,
                                    truncation=True))

        if not self.has_good_form(conv, pred):
            return None

        deal = self.extract_deal(pred, ano2mod)
        return deal

    def get_all_feasible_deals(self, input_seq):
        """Get all feasible deals based on the counts."""
        # first get the counts of the three items in this dialogue.
        cxt = input_seq.split("<history>")[0].replace("<context>","").strip()
        items = cxt.split()
        assert len(items) == 11

        i1_c, i2_c, i3_c = int(items[2].rstrip(",")), int(items[5].rstrip(",")), int(items[8].rstrip(","))
        ccs = [i1_c, i2_c, i3_c]

        issues = None
        if "book" in cxt:
            issues = ["book", "hat", "ball"]
        else:
            issues = ["food", "water", "firewood"]
        
        counts = {}
        for iss, cc in zip(issues, ccs):
            counts[iss] = cc

        mnames = ["<you>","<them>"]
        all_deals = []

        for d1 in range(counts[issues[0]] + 1):
            for d2 in range(counts[issues[1]] + 1):
                for d3 in range(counts[issues[2]] + 1):
                    new_deal = {
                        mnames[0]: {
                            issues[0]: d1,
                            issues[1]: d2,
                            issues[2]: d3,
                        },
                        mnames[1]: {
                            issues[0]: counts[issues[0]] - d1,
                            issues[1]: counts[issues[1]] - d2,
                            issues[2]: counts[issues[2]] - d3,
                        }
                    }
                    all_deals.append(new_deal)
        
        return all_deals

    def get_dd_pred_texts(self, feasible_deals):
        """Convert feasible deals into model outputs.
        always map <you> to <alice>
        """
        batch_pred_texts = []
        for deal in feasible_deals:
            pred_txt = None
            if "book" in deal["<you>"]:
                pred_txt = f"<alice> book={deal['<you>']['book']} hat={deal['<you>']['hat']} ball={deal['<you>']['ball']} <bob> book={deal['<them>']['book']} hat={deal['<them>']['hat']} ball={deal['<them>']['ball']} <EOU>"
            else:
                pred_txt = f"<alice> food={deal['<you>']['food']} water={deal['<you>']['water']} firewood={deal['<you>']['firewood']} <bob> food={deal['<them>']['food']} water={deal['<them>']['water']} firewood={deal['<them>']['firewood']} <EOU>"
            batch_pred_texts.append(pred_txt)

        return batch_pred_texts

    def get_dd_prompt_texts(self, input_seq, cand, num_feasible_deals):
        """Generate prompt texts for a particular candidate utterance."""
        
        if "<selection>" in cand:
            pt_str = input_seq.replace("<history>", f"<history> <you> {cand}")
        else:
            pt_str = input_seq.replace("<history>", f"<history> <them> <selection> <you> {cand}")

        pt_str = pt_str.replace("<you>", "<alice>").replace("<them>", "<bob>")
        
        prompt_texts = [pt_str for _ in range(num_feasible_deals)]
        return prompt_texts

    def get_deal_dists(self, input_seq, cands):
        """Run the predict deal model (batchwise), and compute the entire
        distribution - sorted.
        
        always map <you> to <alice>
        """

        # list of strs, each representing a feasible deal between alice and bob.
        feasible_deals = self.get_all_feasible_deals(input_seq)
        batch_pred_texts = self.get_dd_pred_texts(feasible_deals)

        cand2dist = {}
        for cand in cands:
            
            batch_prompt_texts = self.get_dd_prompt_texts(input_seq, cand, len(feasible_deals))

            encodings = self.tokenizer(
                    batch_prompt_texts, return_tensors="pt", truncation=True, padding=True, max_length=self.max_length
                )

            pred_encodings = self.tokenizer(
                    batch_pred_texts, return_tensors="pt", truncation=True, padding=True, max_length=self.max_length
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
            for ix in range(len(feasible_deals)):
                loss = loss_fct(logits[ix].reshape(-1, self.model.decoder.config.vocab_size), pred_encodings[ix].to(self.device).view(-1))
                scores.append(-loss.to("cpu").item())

            dist = {}
            for deal, score in zip(batch_pred_texts, scores):
                dist[deal.replace("<alice>", "<you>").replace("<bob>","<them>")] = score

            # convert to prob scores
            tot_sum = sum([np.exp(item) for item in dist.values()])
            for deal in dist.keys():
                dist[deal] = np.exp(dist[deal]) / tot_sum

            # sort the deal dict as pr keys.
            sorted_x = sorted(dist.items(), key=operator.itemgetter(1))
            sorted_x.reverse()
            cand2dist[cand] = sorted_x[:]

        return cand2dist