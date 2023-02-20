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

"""
Sup: supervised
Gen: generating the next response
Sel: selection token output
"""
class DealornodealSupGenSel(Dataset):

    def load_raw_dialogues(self):
        """Load raw data from Huggingface."""
        assert not self.raw_data

        hf_dataset = load_dataset("deal_or_no_dialog")
        
        self.raw_data = [dd for dd in hf_dataset[self.split]]

    def get_context(self, agent_input):
        """Get the agent context."""
        
        target_seq = f"book: {agent_input['count'][0]}, {agent_input['value'][0]} hat: {agent_input['count'][1]}, {agent_input['value'][1]}, ball: {agent_input['count'][2]}, {agent_input['value'][2]}"
    
        cxt = f"<context> <target> {target_seq} <persona>"
        
        return cxt
    
    def fix_sent(self, sent):
        """Preprocess the utterance, also add speaker tokens.
        fix <them> or <you>, lowercase, strip.
        """
        sent = sent.replace("YOU:", "<you>").replace("THEM:", "<them>")
        sent = sent.replace("\"","")
        sent = sent.lower().strip()
        return sent

    def get_input_seq(self, agent_context, dialogue):
        """Construct the input sequence."""
        dial2 = dialogue[:]
        dial2.reverse()

        dial2 = " ".join(dial2)

        input_seq = f"{agent_context} <history> {dial2}".strip()

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
            input_seq: your_cxt + reversed history,
            response: your next response
            partner_cxt: partner_cxt for reference.

        For the case of Deal or no deal, the persona will be missing.

        Each raw data corresponds to a dialogue from one perspective.
        So in each case, let's only create processed instances while generating
        then You utterances and not Them utterances.
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
            agent_1_context = self.get_context(dialogue["input"])
            agent_2_context = self.get_context(dialogue["partner_input"])

            sents = dialogue["dialogue"].split("<eos>")

            # Process all utterances in dialogue, only one perspective at a time.
            dialogue_1 = []
            
            for c in sents:
                sentence = self.fix_sent(c)

                if "<you>" in sentence:
                    # then add to processing data.
                    inp_seq = self.get_input_seq(agent_1_context, dialogue_1[:])
                    processed_data.append(f'"{inp_seq}","{sentence}","{agent_2_context}"\n')

                dialogue_1.append(sentence)

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

    def get_count_str(self,):
        """
        get the count string.
        """
        count_str = f"<counts> food=3 water=3 firewood=3"
        return count_str
    
    def fix_sent(self, input):
        """Preprocess the utterance."""
        
        out = input.replace("🙂", "").replace("☹️", "").replace("😮", "").replace("😡", "")
        out = out.replace("\"","")
        out = out.lower().strip()

        return out

    def get_output_seq(self, chat_logs, mapping):
        """Get the output str.

        Order: Food, Water, Firewood
        """
        
        task_data = None
        cid = None
        for ix, c in enumerate(chat_logs):
            if c["text"] == "Submit-Deal":
                assert (ix + 1) == len(chat_logs) - 1
                task_data = c["task_data"]
                cid = c["id"]
                break
                
        assert task_data and cid

        task_dat_alice, task_dat_bob = None, None
        if mapping[cid] == "<alice>":
            task_dat_alice, task_dat_bob = task_data['issue2youget'], task_data['issue2theyget']
        elif mapping[cid] == "<bob>":
            task_dat_alice, task_dat_bob = task_data['issue2theyget'], task_data['issue2youget']
        else:
            raise ValueError

        outp_seq = f"<alice> food={task_dat_alice['Food']} water={task_dat_alice['Water']} firewood={task_dat_alice['Firewood']} <bob> food={task_dat_bob['Food']} water={task_dat_bob['Water']} firewood={task_dat_bob['Firewood']}"
        return outp_seq

    def get_input_seq(self, count_str, dialogue):
        """Construct the input sequence."""
        dial2 = dialogue[:]
        dial2.reverse()

        dial2 = " ".join(dial2)

        input_seq = f"{count_str} <history> {dial2}".strip()

        return input_seq

    def dial_has_exceptions(self, dialogue):

        for c in dialogue["chat_logs"]:
            if c["text"] in ["Reject-Deal", "Walk-Away"]:
                return True
        
        return False

    def process_each_split(self):
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


"""
Sup: supervised
Gen: generating the next response
Sel: selection token output
"""
class CaSiNoSupGenSel(Dataset):

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

    def get_input_seq(self, agent_context, dialogue):
        """Construct the input seq from agent_context and dialogue."""
        
        dial2 = dialogue[:]
        dial2.reverse()
        dial2 = " ".join(dial2)

        input_seq = f"{agent_context} <history> {dial2}".strip()

        return input_seq

    def dial_has_exceptions(self, dialogue):

        for c in dialogue["chat_logs"]:
            if c["text"] in ["Reject-Deal", "Walk-Away"]:
                return True
        
        return False

    def process_each_split(self):
        """Process dialogues in the common format.
        
        For each instance, fill:
            input_seq: your_cxt + reversed history,
            response: your next response
            partner_cxt: partner_cxt for reference.
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
        
        for dialogue in tqdm(all_dialogues):
            
            agent_1_context = self.get_context(dialogue["participant_info"][a1])
            agent_2_context = self.get_context(dialogue["participant_info"][a2])

            dialogue_1 = []
            dialogue_2 = []

            # Process all utterances in dialogue
            for c in dialogue['chat_logs']:
                if c['text'] == 'Submit-Deal':
                    sentence = "<selection>"
                elif c['text'] == 'Accept-Deal':
                    # we don't consider this utterance at all.
                    break
                elif c['text'] == 'Reject-Deal':
                    raise ValueError
                elif c['text'] == 'Walk-Away':
                    raise ValueError
                else:
                    sentence = self.fix_sent(c['text'])
                
                id = c['id']
                if id == a1:
                    inp_seq = self.get_input_seq(agent_1_context, dialogue_1[:])
                    processed_data.append(f'"{inp_seq}","<you> {sentence}","{agent_2_context}"\n')
                    dialogue_1.append(f'<you> {sentence}')
                    dialogue_2.append(f'<them> {sentence}')
                elif id == a2:
                    inp_seq = self.get_input_seq(agent_2_context, dialogue_2[:])
                    processed_data.append(f'"{inp_seq}","<you> {sentence}","{agent_1_context}"\n')
                    dialogue_1.append(f'<them> {sentence}')
                    dialogue_2.append(f'<you> {sentence}')
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

        print("de-duplicated processed data", len(processed_data), len(final_processed_data))

        self.processed_data = final_processed_data[:]