import unittest
import gymnasium

class TestLunarEnv(unittest.TestCase):
    def test_environment_initialization(self):
        lunar_lander = gymnasium.make('LunarLander-v3')
        self.assertIsNotNone(lunar_lander)
        self.assertEqual(8, lunar_lander.observation_space.shape[0])
        print(lunar_lander.observation_space)
        print(dir(lunar_lander))