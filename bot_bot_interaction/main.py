"""
Bot-bot negotiation dialogue.
define agent classes - with a common interface.
and then make them iteract.
"""

# imports
import json
import os
import sys

from config import Config
from game_play import GamePlay

def main():

    # get the config
    config = Config()

    print("All config vars:")
    print("-"*10)
    print(vars(config))
    print("-"*10)

    # setup the results dir.
    if config.override_results:
        os.makedirs(config.results_dir, exist_ok=True)
        
        for f in os.listdir(config.results_dir):
            fpath = os.path.join(config.results_dir, f)
            print(f"Removing file: {f}")
            os.remove(fpath)
    else:
        if os.path.exists(config.results_dir):
            print("Results dir already exists. - Exiting !!!")
            return
        else:
            os.makedirs(config.results_dir)

    # save the config.
    out_path = os.path.join(config.results_dir, "config.json")
    with open(out_path, "w") as f:
        save_config = vars(config)
        del save_config["model_typ2class"]
        json.dump(save_config, f, indent=4)
    print(f"Config stored at: {out_path}")
    
    # get the two agents.
    agents = []
    for model_name, model_typ in zip(config.model_names, config.model_types):
        agents.append(
            config.model_typ2class[model_typ](
                config,
                model_name
                )
            )

    # setup the gameplay object
    game_play = GamePlay(
        config=config,
        agents=agents,
    )

    # perform game play, compute conv specific metrics, and log.
    game_play.game_play()

    # aggregate metrics and log.
    game_play.save_overall_results()

if __name__ == '__main__':
    sys.exit(main())