
from catanatron import Game, Player, RandomPlayer, Color

class RuleBot(Player):

    def decide(self, game, playable_actions):
        # Smaller number = higher priority
        priority = {
            "BUILD_CITY": 1,
            "BUILD_SETTLEMENT": 2,
            "BUY_DEVELOPMENT_CARD": 3,
            "BUILD_ROAD": 4,
            "ROLL": 5,
            "END_TURN": 100
        }

        actions = list(playable_actions)

        return min(
            actions,
            key=lambda action: priority.get(
                action[0].name, 50
            )
        )


players = [
    RuleBot(Color.RED),
    RandomPlayer(Color.BLUE),
    RandomPlayer(Color.WHITE),
    RandomPlayer(Color.ORANGE),
]

game = Game(players)
print("Winner:", game.play())
