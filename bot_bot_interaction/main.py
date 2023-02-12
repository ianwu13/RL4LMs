"""
Bot-bot negotiation dialogue.
define agent classes - with a common interface.
and then make them iteract.
"""

# imports
import os
import sys
import glob

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
    for ix, model_typ in enumerate(config.model_types):
        agents.append(config.model_typ2class[model_typ](ix, config))

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