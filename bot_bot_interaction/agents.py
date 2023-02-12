import os
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM # type: ignore

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

        model_path = os.path.join(self.config.log_dir, self.name, "model")
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_path)

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
        self.input_seq = f"<context> {cxt} <history>"
    
    def receive(self, resp_obj):
        """Receive a response object."""
        assert resp_obj.name != self.name # came from the opponent

        # add to the start of the input_seq.
        self.input_seq = self.input_seq.replace("<history>", f"<history> <them> {resp_obj['resp']}")

    def respond(self):
        """Respond to the input request."""
        
        input_ids = self.tokenizer(self.input_seq, return_tensors="pt", truncation=True, padding=True, max_length=self.input_max_length).input_ids
        out_encs = self.model.generate(
            input_ids,
            num_beams=self.num_beams,
            do_sample=self.do_sample, 
            top_p=self.top_p,
            top_k=self.top_k,
            max_new_tokens=self.max_new_tokens,
            min_length=self.min_length
            )
        resp = self.tokenizer.decode(out_encs[0],
                                        max_length=self.max_new_tokens,
                                        truncation=True)
        resp = resp.replace("<unk>you>", "").replace("<unk>EOU>", "").strip()
        resp = resp.replace("<pad>", "").replace("</s>", "").strip()

        # add response to the start of the history.
        self.input_seq = self.input_seq.replace("<history>", f"<history> <you> {resp}")

        resp_obj = {
            "resp": resp,
            "name": self.name,
        }

        return resp_obj

class OfflineRLAgent(Agent):
    def __init__(self, config, name) -> None:
        super().__init__(config, name)

        self.initial_rtg = 50

        tokenizer_id = "t5-base"
        padding_side = "right"
        truncation_side = "right"
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_id)
        self.tokenizer.truncation_side = truncation_side
        self.tokenizer.padding_side = padding_side

        model_path = os.path.join(self.config.log_dir, self.name, "model")
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_path)

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
        assert resp_obj.name != self.name # came from the opponent

        # add to the start of the input_seq.
        self.input_seq = self.input_seq.replace("<history>", f"<history> <them> {resp_obj['resp']}")

    def respond(self):
        """Respond to the input request."""
        
        input_ids = self.tokenizer(self.input_seq, return_tensors="pt", truncation=True, padding=True, max_length=self.input_max_length).input_ids
        out_encs = self.model.generate(
            input_ids,
            num_beams=self.num_beams,
            do_sample=self.do_sample, 
            top_p=self.top_p,
            top_k=self.top_k,
            max_new_tokens=self.max_new_tokens,
            min_length=self.min_length
            )
        resp = self.tokenizer.decode(out_encs[0],
                                        max_length=self.max_new_tokens,
                                        truncation=True)
        resp = resp.replace("<unk>you>", "").replace("<unk>EOU>", "").strip()
        resp = resp.replace("<pad>", "").replace("</s>", "").strip()

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