import argparse

from Game import generals_game

# this starts up the generals environment and begins the game
parser = argparse.ArgumentParser(description="Three-player Generals-style AI environment")
parser.add_argument(
    "--seed",
    type=int,
    default=None,
    help="Optional reproducible map seed",
)

arguments = parser.parse_args()
application = generals_game.GeneralsApp(seed=arguments.seed)
application.run()
