import os
from tqdm import tqdm
from datasets import load_dataset

class Dataset:
    def __init__(self, dpath, split):
        self.dpath = dpath
        self.split = split
        self.raw_data = None

        # load data if available
        self.opath = os.path.join(self.dpath, f"{self.split}.csv")
        self.processed_data = None
        if os.path.exists(self.opath):
            self.processed_data = load_dataset("csv", data_files={self.split: self.opath})[self.split]

    def save_processed_instances(self):
        """Save processed data to csv files."""
        assert self.processed_data

        out_file = open(self.opath, 'w')
        out_file.write('input_seq,response,partner_cxt\n')

        for out_str in self.processed_data:
            out_file.write(out_str)

    def print_stats(self):
        """Print basic data stats."""
        
        assert self.processed_data

        print("-"*10)
        print(f"Dataset opath: {self.opath}")
        if self.raw_data:
            print(f"Raw: {len(self.raw_data)}")
        print(f"Processed: {len(self.processed_data)}")
        print("-"*10)

    def load_raw_dialogues(self):
        """Load raw data from Huggingface."""
        raise NotImplementedError

    def process_each_split(self):
        """Process each split data."""
        raise NotImplementedError

    def load_dataset(self):
        """
        Primary method to load the dataset, given a split name.
        """

        if self.processed_data:
            print("Already processed data found. Directly using that.")
            self.print_stats()
            return self.processed_data
        
        self.load_raw_dialogues()

        self.process_each_split()

        self.save_processed_instances()

        self.print_stats()

        # load data from file in dict format.
        assert os.path.exists(self.opath)
        data_dicts = load_dataset("csv", data_files={self.split: self.opath})[self.split]

        return data_dicts

class DealornodealPredictAgreedDeal(Dataset):

    def load_raw_dialogues(self):
        """Load raw data from Huggingface."""
        assert not self.raw_data

        hf_dataset = load_dataset("deal_or_no_dialog")
        
        self.raw_data = [dd for dd in hf_dataset[self.split]]

    def get_count_str(self, agent_input):
        """
        get the count string.
        """
        count_str = f"<counts> book={agent_input['count'][0]} hat={agent_input['count'][1]} ball={agent_input['count'][2]}"

        return count_str

    def get_output_seq(self, d_output, mapping):
        """Get the output str.

        Order: Books, Hats, Balls
        """
        assert "<disconnect>" not in d_output
        assert "<no_agreement>" not in d_output
        assert "<disagree" not in d_output

        deal_items = [int(ii.split("=")[-1]) for ii in d_output.split()]

        if "<alice>" == mapping["YOU:"]:
            outp_seq = f"<alice> book={deal_items[0]} hat={deal_items[1]} ball={deal_items[2]} <bob> book={deal_items[3]} hat={deal_items[4]} ball={deal_items[5]}"
        elif "<alice>" == mapping["THEM:"]:
            outp_seq = f"<alice> book={deal_items[3]} hat={deal_items[4]} ball={deal_items[5]} <bob> book={deal_items[0]} hat={deal_items[1]} ball={deal_items[2]}"
        else:
            raise ValueError

        return outp_seq
    
    def fix_sent(self, sent, mapping):
        """Preprocess the utterance, also add speaker tokens.
        fix <them> or <you>, lowercase, strip.
        """
        sent = sent.replace("YOU:", mapping["YOU:"]).replace("THEM:", mapping["THEM:"])
        sent = sent.replace("\"","")
        sent = sent.lower().strip()
        return sent

    def get_input_seq(self, count_str, dialogue):
        """Construct the input sequence."""
        dial2 = dialogue[:]
        dial2.reverse()

        dial2 = " ".join(dial2)

        input_seq = f"{count_str} <history> {dial2}".strip()

        return input_seq

    def dial_has_exceptions(self, dial):
        output = dial["output"]
        if "<disagree>" in output or "<disconnect>" in output or "<no_agreement>" in output:
            # ignore these cases
            return True
        return False

    def process_each_split(self):
        """Process dialogues in the common format.
        
        For each instance, fill:
            input_seq: item counts + reversed history (alice / bob),
            response: deal counts for alice and bob
            partner_cxt: partner_cxt for reference - who cares.

        For DND, each raw data corresponds to a dialogue from one perspective.
        
        For each instance, we will assign Alice and Bob in both the possible ways. Then at the end, we remove duplicates.

        Ultimately, from each dialogue, we only contain two instances.

        """

        all_dialogues = []

        print("Raw data: ", len(self.raw_data))
        for dial in self.raw_data:
            # if the negotiation did not reach agreement or contains reject sequences, ignore
            if self.dial_has_exceptions(dial):
                continue
            all_dialogues.append(dial)
        print("remove exceptions", len(all_dialogues))

        processed_data = []
        
        for dialogue in tqdm(all_dialogues):
            count_str = self.get_count_str(dialogue["input"])

            sents = dialogue["dialogue"].split("<eos>")
            
            mappings = [
                {
                    "YOU:": "<alice>",
                    "THEM:": "<bob>",
                },
                {
                    "THEM:": "<alice>",
                    "YOU:": "<bob>",
                },
            ]

            for mapping in mappings:
                # Process all utterances in dialogue based on the mapping.
                dialogue_1 = [self.fix_sent(c, mapping) for c in sents]

                inp_seq = self.get_input_seq(count_str, dialogue_1[:])
                outp_seq = self.get_output_seq(dialogue["output"], mapping)
                processed_data.append(f'"{inp_seq}","{outp_seq}","dummy"\n')

        # remove duplicates
        final_processed_data = []
        pd_set = set()
        for pd in processed_data:
            if pd in pd_set:
                continue
            final_processed_data.append(pd)
            pd_set.add(pd)

        print("de-duplicated processed data", len(processed_data), len(final_processed_data))

        self.processed_data = final_processed_data[:]


class CaSiNoPredictAgreedDeal(Dataset):

    def load_raw_dialogues(self):
        """Load raw data from Huggingface."""
        assert not self.raw_data

        hf_dataset = load_dataset("casino", split="train")
        
        self.raw_data = [dd for dd in hf_dataset]
        assert len(self.raw_data) == 1030, len(self.raw_data)

        #fix as per self.split
        if self.split == "train":
            self.raw_data = self.raw_data[:int(0.8*len(self.raw_data))]
        elif self.split == "validation":
            self.raw_data = self.raw_data[int(0.8*len(self.raw_data)):int(0.9*len(self.raw_data))]
        elif self.split == "test":
            self.raw_data = self.raw_data[int(0.9*len(self.raw_data)):]
        else:
            raise ValueError

    def get_context(self, agent_info):
        """Get context string."""
        
        agent_pref = agent_info["value2issue"]
        agent_reasons = agent_info['value2reason']

        pref_score = {
            "food": None,
            "water": None,
            "firewood": None,
        }

        pref_score[agent_pref["High"].lower()] = 5
        pref_score[agent_pref["Medium"].lower()] = 4
        pref_score[agent_pref["Low"].lower()] = 3

        target_seq = f"food: 3, {pref_score['food']} water: 3, {pref_score['water']} firewood: 3, {pref_score['firewood']}"

        hp_sentence = f'my highest priority is {agent_pref["High"]} because {agent_reasons["High"]}'
        mp_sentence = f'my medium priority is {agent_pref["Medium"]} because {agent_reasons["Medium"]}'
        lp_sentence = f'my lowest priority is {agent_pref["Low"]} because {agent_reasons["Low"]}'
    
        cxt = f"<context> <target> {target_seq} <persona> {self.fix_sent(hp_sentence)} {self.fix_sent(mp_sentence)} {self.fix_sent(lp_sentence)}"
        
        return cxt
    
    def fix_sent(self, input):
        """Preprocess the utterance."""
        
        out = input.replace("🙂", "").replace("☹️", "").replace("😮", "").replace("😡", "")
        out = out.replace("\"","")
        out = out.lower().strip()

        return out
    
    def get_submit_deal_sentence(self, task_data):
        """Get submit deal sentence."""
        task_dat_you = task_data['issue2youget']
        task_dat_them = task_data['issue2theyget']
        return f"let's submit this deal. i get {task_dat_you['Food']} food, {task_dat_you['Water']} water, and {task_dat_you['Firewood']} firewood. you get {task_dat_them['Food']} food, {task_dat_them['Water']} water, and {task_dat_them['Firewood']} firewood."

    def get_input_seq(self, agent_context, dialogue, rtg_seq):
        """Construct the input seq from agent_context and dialogue."""
        
        dial2 = dialogue[:]
        dial2.reverse()
        dial2 = "".join(dial2)

        # set up rtgs - get the number of times <you> appears in the current
        # dialogue and then add 1 -> get these many elements from the rtg_seq
        req_rtgs_count = dial2.count("<you>") + 1
        req_rtgs = rtg_seq[:req_rtgs_count]
        req_rtgs = " ".join([str(ii) for ii in req_rtgs])

        input_seq = f"<rewards> {req_rtgs} {agent_context} <history>{dial2}".strip()

        return input_seq

    def get_total_points(self, dialogue, aid):
        """Get the total points scored by aid agent."""
        
        agent_info = dialogue["participant_info"][aid]
        agent_pref = agent_info["value2issue"]

        pref_score = {
            "food": None,
            "water": None,
            "firewood": None,
        }

        pref_score[agent_pref["High"].lower()] = 5
        pref_score[agent_pref["Medium"].lower()] = 4
        pref_score[agent_pref["Low"].lower()] = 3

        # get the deal.
        my_deal = {
            "food": -1,
            "water": -1,
            "firewood": -1,
        }

        save_ix = -1
        for ix, item in enumerate(dialogue["chat_logs"]):
            if item["text"] == "Submit-Deal":
                save_ix = ix
        
        assert save_ix != -1
        assert dialogue["chat_logs"][save_ix + 1]["text"] == "Accept-Deal"
        assert dialogue["chat_logs"][save_ix]["text"] == "Submit-Deal"
        assert (save_ix + 1) == (len(dialogue["chat_logs"]) - 1)

        # agreement was reached, and save_ix contains the deal.
        deal_key = "issue2youget"
        if aid != dialogue["chat_logs"][save_ix]["id"]:
            deal_key = "issue2theyget"
        
        deal_data = dialogue["chat_logs"][save_ix]["task_data"][deal_key]
        my_deal["food"] = int(deal_data["Food"])
        my_deal["water"] = int(deal_data["Water"])
        my_deal["firewood"] = int(deal_data["Firewood"])

        total_points = my_deal["food"]*pref_score["food"] + my_deal["water"]*pref_score["water"] + my_deal["firewood"]*pref_score["firewood"]
        return total_points

    def get_rtg_seq(self, dialogue, aid, split):
        """Get the rtg sequence."""
        
        total_points = self.get_total_points(dialogue, aid)

        rewards = []
        for c in dialogue["chat_logs"]:
            if c["id"] != aid:
                continue
            
            this_rew = 0
            
            # for a new message
            this_rew -= 2

            rewards.append(this_rew)

        if split == "train":
            
            if aid == dialogue["chat_logs"][-1]["id"]:
                # this player makes the deal decision.
                if dialogue["dial_type"] in ["cs_cd", "ws_cd"]:
                    rewards[-1] += total_points*3
            else:
                # this player submits the deal.
                if dialogue["dial_type"] in ["cs_cd", "cs_wd"]:
                    rewards[-1] += total_points*3
        else:
            rewards[-1] += total_points*3

        rtgs = []
        for i in range(len(rewards)):
            rtgs.append(sum(rewards[i:]))

        return rtgs

    def dial_has_exceptions(self, dialogue):

        for c in dialogue["chat_logs"]:
            if c["text"] in ["Reject-Deal", "Walk-Away"]:
                return True
        
        return False

    def add_perturb_data(self, dialogues):

        new_dialogues = dialogues[:]
        for ix in range(len(new_dialogues)):
            new_dialogues[ix]["dial_type"] = "cs_cd" # correct submit and correct decision.
        
        # correct submit and wrong decision.
        for c in new_dialogues:

            if c["dial_type"] != "cs_cd":
                continue

            assert c["chat_logs"][-1]["text"] == "Accept-Deal"

            c_pert = copy.deepcopy(c)
            c_pert["chat_logs"][-1]["text"] == "Reject-Deal"
            c_pert["dial_type"] = "cs_wd"

            new_dialogues.append(c_pert)

        # wrong deal and wrong decision
        for c in new_dialogues:

            if c["dial_type"] != "cs_cd":
                continue

            assert c["chat_logs"][-2]["text"] == "Submit-Deal"

            while True:
                c_pert = copy.deepcopy(c)

                for issue in ["Food", "Water", "Firewood"]:
                    assert c_pert["chat_logs"][-2]["task_data"]["issue2youget"][issue]
                    
                    if random.uniform(0, 1) < 0.5:
                        wrong_opts = []
                        for opt in ["0", "1", "2", "3"]:
                            if opt != c_pert["chat_logs"][-2]["task_data"]["issue2youget"][issue]:
                                wrong_opts.append(opt)
                    
                        wrong_opt_selected = random.choice(wrong_opts)
                        assert wrong_opt_selected != c_pert["chat_logs"][-2]["task_data"]["issue2youget"][issue]
                        c_pert["chat_logs"][-2]["task_data"]["issue2youget"][issue] = wrong_opt_selected
                        c_pert["chat_logs"][-2]["task_data"]["issue2theyget"][issue] = str(3 - int(wrong_opt_selected))
                
                f = 0
                for issue in ["Food", "Water", "Firewood"]:
                    if c_pert["chat_logs"][-2]["task_data"]["issue2youget"][issue] != c["chat_logs"][-2]["task_data"]["issue2youget"][issue]:
                        f = 1
                        break
                
                if f:
                    break
            
            c_pert["dial_type"] = "ws_wd"
            new_dialogues.append(c_pert)

        # wrong deal and correct decision
        for c in new_dialogues:

            if c["dial_type"] != "cs_cd":
                continue

            assert c["chat_logs"][-2]["text"] == "Submit-Deal"

            while True:
                c_pert = copy.deepcopy(c)

                for issue in ["Food", "Water", "Firewood"]:
                    assert c_pert["chat_logs"][-2]["task_data"]["issue2youget"][issue]
                    
                    if random.uniform(0, 1) < 0.5:
                        wrong_opts = []
                        for opt in ["0", "1", "2", "3"]:
                            if opt != c_pert["chat_logs"][-2]["task_data"]["issue2youget"][issue]:
                                wrong_opts.append(opt)
                    
                        wrong_opt_selected = random.choice(wrong_opts)
                        assert wrong_opt_selected != c_pert["chat_logs"][-2]["task_data"]["issue2youget"][issue]
                        c_pert["chat_logs"][-2]["task_data"]["issue2youget"][issue] = wrong_opt_selected
                        c_pert["chat_logs"][-2]["task_data"]["issue2theyget"][issue] = str(3 - int(wrong_opt_selected))
                
                f = 0
                for issue in ["Food", "Water", "Firewood"]:
                    if c_pert["chat_logs"][-2]["task_data"]["issue2youget"][issue] != c["chat_logs"][-2]["task_data"]["issue2youget"][issue]:
                        f = 1
                        break
                
                if f:
                    break
            
            assert c["chat_logs"][-1]["text"] == "Accept-Deal"
            c_pert["chat_logs"][-1]["text"] == "Reject-Deal"

            c_pert["dial_type"] = "ws_cd"
            new_dialogues.append(c_pert)

        return new_dialogues

    def process_each_split(self, split):
        """Process dialogues in the common format.
        
        For each instance, fill:
            input_seq: item counts + reversed history (alice / bob),
            response: deal counts for alice and bob
            partner_cxt: partner_cxt for reference - who cares.

        For Casino, each raw data corresponds to a dialogue from both the perspectives.
        
        For each instance, we will assign Alice and Bob in both the possible ways. Then at the end, we remove duplicates (although in this case, there should not be any).

        Ultimately, from each dialogue, we only contain two instances.

        """
        
        all_dialogues = []
        
        print("Raw data: ", len(self.raw_data))
        for dial in self.raw_data:
            # if the negotiation did not reach agreement or contains reject sequences, ignore
            if self.dial_has_exceptions(dial):
                continue
            all_dialogues.append(dial)
        print("remove exceptions", len(all_dialogues))

        processed_data = []

        a1 = "mturk_agent_1"
        a2 = "mturk_agent_2"

        mappings = [
                {
                    a1: "<alice>",
                    a2: "<bob>",
                },
                {
                    a2: "<alice>",
                    a1: "<bob>",
                },
            ]
        
        for dialogue in tqdm(all_dialogues):
            
            count_str = self.get_count_str()

            for mapping in mappings:
                dialogue_1 = []

                # Process all utterances in dialogue
                for c in dialogue['chat_logs']:
                    assert c['text'] not in ['Accept-Deal', 'Reject-Deal', 'Walk-Away']
                    
                    mid = mapping[c["id"]]

                    if c['text'] == 'Submit-Deal':
                        sentence = f"{mid} <selection>"
                        dialogue_1.append(sentence)
                        break
                        
                    sentence = self.fix_sent(c['text'])
                    sentence = f"{mid} {sentence}"
                    dialogue_1.append(sentence)

                inp_seq = self.get_input_seq(count_str, dialogue_1[:])
                outp_seq = self.get_output_seq(dialogue["chat_logs"], mapping)
                processed_data.append(f'"{inp_seq}","{outp_seq}","dummy"\n')
                    
                if id == a1:
                    inp_seq = self.get_input_seq(agent_1_context, dialogue_1[:], rtg_seq_1[:])
                    processed_data.append(f'"{inp_seq}","<you> {sentence}","{agent_2_context}"\n')
                    dialogue_1.append(f' <you> {sentence}')
                    dialogue_2.append(f' <them> {sentence}')
                elif id == a2:
                    inp_seq = self.get_input_seq(agent_2_context, dialogue_2[:], rtg_seq_2[:])
                    processed_data.append(f'"{inp_seq}","<you> {sentence}","{agent_1_context}"\n')
                    dialogue_1.append(f' <them> {sentence}')
                    dialogue_2.append(f' <you> {sentence}')
                else:
                    raise Exception('INVALID AGENT ID')

        # remove duplicates
        final_processed_data = []
        pd_set = set()
        for pd in processed_data:
            if pd in pd_set:
                continue
            final_processed_data.append(pd)
            pd_set.add(pd)

        print("dups", len(processed_data), len(final_processed_data))
        return final_processed_data


        DealornodealPredictAgreedDeal

        all_dialogues = []

        print("Raw data: ", len(self.raw_data))
        for dial in self.raw_data:
            # if the negotiation did not reach agreement or contains reject sequences, ignore
            if self.dial_has_exceptions(dial):
                continue
            all_dialogues.append(dial)
        print("remove exceptions", len(all_dialogues))

        processed_data = []
        
        for dialogue in tqdm(all_dialogues):
            count_str = self.get_count_str(dialogue["input"])

            sents = dialogue["dialogue"].split("<eos>")
            
            mappings = [
                {
                    "YOU:": "<alice>",
                    "THEM:": "<bob>",
                },
                {
                    "THEM:": "<alice>",
                    "YOU:": "<bob>",
                },
            ]

            for mapping in mappings:
                # Process all utterances in dialogue based on the mapping.
                dialogue_1 = [self.fix_sent(c, mapping) for c in sents]

                inp_seq = self.get_input_seq(count_str, dialogue_1[:])
                outp_seq = self.get_output_seq(dialogue["output"], mapping)
                processed_data.append(f'"{inp_seq}","{outp_seq}","dummy"\n')

        # remove duplicates
        final_processed_data = []
        pd_set = set()
        for pd in processed_data:
            if pd in pd_set:
                continue
            final_processed_data.append(pd)
            pd_set.add(pd)

        print("de-duplicated processed data", len(processed_data), len(final_processed_data))

        self.processed_data = final_processed_data[:]