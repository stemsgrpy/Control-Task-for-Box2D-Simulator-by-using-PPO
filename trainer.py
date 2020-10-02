import math
import numpy as np
from config import Config
from core.logger import TensorBoardLogger
from core.util import get_output_folder

class Trainer:
    def __init__(self, agent, env, config: Config):
        self.agent = agent
        self.env = env
        self.config = config

        self.outputdir = get_output_folder(self.config.output, self.config.env)
        self.agent.save_config(self.outputdir)
        self.board_logger = TensorBoardLogger(self.outputdir)

    def train(self, pre_fr=0):
        all_rewards = []
        tmp_reward = 0
        episode_reward = 0
        ep_num = 0
        is_win = False

        state = self.env.reset()

        for fr in range(pre_fr + 1, self.config.frames + 1):
            # self.env.render()
            action = self.agent.act(state)

            next_state, reward, done, _ = self.env.step(action)
            reward = float(reward)

            self.agent.buffer.rewards.append(reward)
            self.agent.buffer.is_terminals.append(done)

            state = next_state
            episode_reward += reward

            if fr % self.config.update_tar_interval == 0:
                self.agent.learning(fr)
                self.agent.buffer.clear_memory()

            if fr % self.config.print_interval == 0:
                print("frames: %5d, reward: %5f, episode: %4d" % (fr, np.mean(all_rewards[-10:]), ep_num))

            if fr % self.config.log_interval == 0:
                self.board_logger.scalar_summary('Reward per episode', ep_num, all_rewards[-1])

            if self.config.checkpoint and fr % self.config.checkpoint_interval == 0:
                self.agent.save_checkpoint(fr, self.outputdir)

            if done:
                state = self.env.reset()

                all_rewards.append(episode_reward)
                episode_reward = 0
                ep_num += 1
                avg_reward = float(np.mean(all_rewards[-100:]))
                self.board_logger.scalar_summary('Best 100-episodes average reward', ep_num, avg_reward)

                if len(all_rewards) >= 100 and avg_reward >= self.config.win_reward and all_rewards[-1] > self.config.win_reward:
                    is_win = True
                    self.agent.save_model(self.outputdir, 'best')
                    print('Ran %d episodes best 100-episodes average reward is %3f. Solved after %d trials ✔' % (ep_num, avg_reward, ep_num - 100))
                    if self.config.win_break:
                        break
                elif len(all_rewards) >= 100 and avg_reward > tmp_reward:
                    tmp_reward = avg_reward
                    self.agent.save_model(self.outputdir, 'tmp')
                    print('Ran %d episodes tmp 100-episodes average reward is %3f. tmp Solved after %d trials' % (ep_num, avg_reward, ep_num - 100))

        if not is_win:
            print('Did not solve after %d episodes' % ep_num)
            self.agent.save_model(self.outputdir, 'last')
