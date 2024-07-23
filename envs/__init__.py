import gymnasium
from envs.darkroom import DarkRoom


gymnasium.register(id="DarkRoom", entry_point=DarkRoom)
