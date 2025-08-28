import numpy as np
import torch
from pathlib import Path
import json
from datetime import datetime
import subprocess
import time
import webbrowser

from environments import LunarEnvironment
from agents.dqn import DQNAgent, DQNUpdater
from utils.logger import Logger





if __name__ == "__main__":
    train_dqn(episodes=1000)