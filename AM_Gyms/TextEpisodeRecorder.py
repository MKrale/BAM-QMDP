import os
import gymnasium as gym
from gymnasium.utils.save_video import capped_cubic_video_schedule


class TextEpisodeRecorder(gym.Wrapper):

    def __init__(
        self,
        env,
        folder: str = "./",
        episode_trigger=capped_cubic_video_schedule,
        name_prefix: str = "training",
    ):
        super().__init__(env)

        self.folder = folder
        self.episode_trigger = episode_trigger
        self.name_prefix = name_prefix

        self.recording = False
        self.episode = -1
        self.file = None

    def reset(self, seed=None, options=None):
        obs, info = self.env.reset(seed=seed, options=options)
        self.episode += 1

        if self.recording:
            self.stop_recording()
        if self.episode_trigger(self.episode):
            self.start_recording()

        return obs, info

    def step(self, action):
        obs, rew, terminated, truncated, info = self.env.step(action)
        if self.recording:
            self.record_frame()
        return obs, rew, terminated, truncated, info

    def record_frame(self):
        text = self.env.render()
        self.file.write(text)

    def stop_recording(self):
        self.file.close()
        self.recording = False

    def start_recording(self):
        full_file_name = (
            self.folder + "/" + self.name_prefix + str(self.episode) + ".txt"
        )
        # Ensure the folder exists
        os.makedirs(self.folder, exist_ok=True)

        # initialize file, overwrite if it exists
        self.file = open(full_file_name, "w")
        self.file.close()

        # now open it in append mode so we can append at every action
        self.file = open(full_file_name, "a")

        self.recording = True
