import tensorflow as tf
import matplotlib.pyplot as plt

def compute_saliency(model, observation):
    with tf.GradientTape() as tape:
        tape.watch(observation)
        prediction = model(observation)
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