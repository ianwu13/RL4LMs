"""
Bot-bot negotiation dialogue.

define agent classes - with a common interface.

and then make them iteract.

"""

# imports
import os
import sys
import glob
import random

class Agent:
    def __init__(self) -> None:
        """Load the model, tokenizer, define any internal states."""  
        raise NotImplementedError

    def respond(self, request):
        """Respond to the input json request."""
        raise NotImplementedError

class SupervisedAgent(Agent):
    def __init__(self) -> None:
        super().__init__()
        pass

class OfflineRLAgent(Agent):
    def __init__(self) -> None:
        super().__init__()
        pass

class GamePlay:
    def __init__(self, config, agents) -> None:
        """Get the two model objects, maintain internal states."""
        self.config = config
        self.agents = agents

    def game_play(self):
        """Make the two models interact with each other and store the logs."""
        
        convs = []

        ag_cxts = self.choose_agent_contexts()

        # setup agent contexts - and reset internal storage.
        for ag, ag_cxt in zip(self.agents, ag_cxts):
            ag.reset_agent(ag_cxt)

        for conv_ix in range(self.config.num_convs):
            print(f"Starting conv: {conv_ix}")

            this_conv = []
            curr_ag = random.choice([0,1])
            for _ in range(self.config.max_utts):
                
                new_resp = self.agents[curr_ag].respond()
                resp_obj = {
                    "resp": new_resp,
                    "ag_ix": self.agents[curr_ag].name,
                }

                this_conv.append(resp_obj)

                if self.conv_done(this_conv):
                    break

                curr_ag = 1 - curr_ag

            convs.append(this_conv)

        

    def results(self):
        """Compute the results from the interactions and store them in a file."""
        pass

class Config:
    def __init__(self) -> None:

        # I/O
        self.log_dir = ""
        self.out_dir = ""
        self.model_names = [

        ]
        self.model_types = ["offline_rl", "supervised"]
        self.override_results = True

        self.model_typ2class = {
            "offline_rl": OfflineRLAgent,
            "supervised": SupervisedAgent,
        }
    
        # tokenizer params

        # interaction
        self.num_convs = 10 # total conversations to be logged.
        self.max_utts = 20 # max utterances in one conversation - hard stop.


        # generate params 

        # process
        self.results_dir = os.path.join(self.out_dir, f"{self.model1}_{self.model2}")

def main():

    # get the config
    config = Config()

    # setup the results dir.
    if config.override_results:
        os.makedirs(config.results_dir, exist_ok=True)
        
        files = glob.glob(config.results_dir)
        for f in files:
            os.remove(f)
    else:
        if os.path.exists(config.results_dir):
            print("Results dir already exists. - Exiting !!!")
            return
        else:
            os.makedirs(config.results_dir)
    
    # get the two agents.
    agents = []
    for model_typ in config.model_types:
        agents.append(config.model_typ2class[model_typ]())

    # setup the gameplay object
    game_play = GamePlay(
        config=config,
        agents=agents,
    )

    # perform game play and log.
    game_play.game_play()

    # compute results and log.
    game_play.results()


if __name__ == '__main__':
    sys.exit(main())