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

        assert self.processed_data
        return self.processed_data

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