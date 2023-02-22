"""
Class to use the trained model for predicting the agreed deal from a finished conversation.
"""

import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM # type: ignore

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
        alice_stuff, bob_stuff = pred.split("<bob>")[0].strip(), pred.split("<bob>")[-1].strip()

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