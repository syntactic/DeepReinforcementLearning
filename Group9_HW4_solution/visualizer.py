from IPython.display import HTML
from base64 import b64encode
import gymnasium as gym
from gymnasium.utils.save_video import save_video
import tensorflow as tf
import numpy as np

def play(filename):
    html = ''
    video = open(filename,'rb').read()
    src = 'data:video/mp4;base64,' + b64encode(video).decode()
    html += '<video width=600 controls autoplay loop><source src="%s" type="video/mp4"></video>' % src
    return HTML(html)

def run_episodes_and_save_videos(num_episodes: int, model_path: str):
    env = gym.make("LunarLander-v2", render_mode="rgb_array_list")
    _ = env.reset()
    episode_index = 0
    model = tf.keras.models.load_model(model_path)

    for episode_index in range(num_episodes):
        state, _ = env.reset()

        while True:
            q_values = model(np.array([state,]))
            action = tf.argmax(q_values[0]).numpy()

            next_state, reward, terminated, truncated, _ = env.step(action)

            if terminated or truncated:
                save_video(
                    env.render(),
                    "videos",
                    fps=env.metadata["render_fps"],
                    episode_index=episode_index,
                    episode_trigger=lambda x : True
                  )
                break
    env.close()
