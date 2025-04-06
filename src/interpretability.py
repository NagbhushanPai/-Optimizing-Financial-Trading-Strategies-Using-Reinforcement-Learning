import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np

def compute_saliency(model, observation):
    observation = tf.convert_to_tensor(observation, dtype=tf.float32)
    with tf.GradientTape() as tape:
        tape.watch(observation)
        prediction = model.predict(observation)[0]  # Adjust based on model output
    gradients = tape.gradient(prediction, observation)
    saliency = tf.reduce_max(tf.abs(gradients), axis=0)
    return saliency.numpy()

def plot_saliency(saliency, feature_names):
    plt.figure(figsize=(10, 6))
    plt.bar(range(len(saliency)), saliency)
    plt.xticks(range(len(saliency)), feature_names, rotation=45)
    plt.title("Saliency Map for Trading Decision")
    plt.xlabel("Feature")
    plt.ylabel("Importance")
    plt.tight_layout()
    plt.show()

# Example usage (to be called after training)
def analyze_interpretability(model, env):
    obs = env.reset()
    saliency = compute_saliency(model, obs)
    feature_names = env.data.columns.tolist() if hasattr(env.data, 'columns') else [f"Feature_{i}" for i in range(len(obs))]
    plot_saliency(saliency, feature_names)