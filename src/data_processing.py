import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
import pywt
import tensorflow as tf
from tensorflow.keras import layers, Model

class WGAN_GP:
    def __init__(self, latent_dim=100, feature_dim=5):
        self.latent_dim = latent_dim
        self.feature_dim = feature_dim
        self.generator = self.build_generator()
        self.critic = self.build_critic()
        self.generator.optimizer = tf.keras.optimizers.Adam(learning_rate=0.0001, beta_1=0.5, beta_2=0.9)
        self.critic.optimizer = tf.keras.optimizers.Adam(learning_rate=0.0001, beta_1=0.5, beta_2=0.9)

    def build_generator(self):
        model = tf.keras.Sequential([
            layers.Dense(128, activation='relu', input_shape=(self.latent_dim,)),
            layers.Dense(256, activation='relu'),
            layers.Dense(self.feature_dim, activation='tanh')
        ])
        return model

    def build_critic(self):
        model = tf.keras.Sequential([
            layers.Dense(128, activation='relu', input_shape=(self.feature_dim,)),
            layers.Dense(256, activation='relu'),
            layers.Dense(1)
        ])
        return model

    def gradient_penalty(self, real, fake):
        alpha = tf.random.uniform([tf.shape(real)[0], 1], 0., 1.)
        diff = fake - real
        interp = real + alpha * diff
        with tf.GradientTape() as tape:
            tape.watch(interp)
            pred = self.critic(interp, training=True)
        grads = tape.gradient(pred, [interp])[0]
        norm = tf.sqrt(tf.reduce_sum(tf.square(grads), axis=[1]))
        gp = tf.reduce_mean((norm - 1.)**2)
        return gp

    def train(self, real_data, epochs=100, batch_size=32):
        for epoch in range(epochs):
            for i in range(0, len(real_data), batch_size):
                batch = real_data[i:i+batch_size]
                noise = tf.random.normal([batch_size, self.latent_dim])

                with tf.GradientTape() as tape:
                    fake_data = self.generator(noise, training=True)
                    critic_real = self.critic(batch, training=True)
                    critic_fake = self.critic(fake_data, training=True)
                    gp = self.gradient_penalty(batch, fake_data)
                    critic_loss = tf.reduce_mean(critic_fake) - tf.reduce_mean(critic_real) + gp * 10.0

                critic_grads = tape.gradient(critic_loss, self.critic.trainable_variables)
                self.critic.optimizer.apply_gradients(zip(critic_grads, self.critic.trainable_variables))

                if i % 5 == 0:
                    with tf.GradientTape() as tape:
                        fake_data = self.generator(noise, training=True)
                        critic_fake = self.critic(fake_data, training=True)
                        gen_loss = -tf.reduce_mean(critic_fake)

                    gen_grads = tape.gradient(gen_loss, self.generator.trainable_variables)
                    self.generator.optimizer.apply_gradients(zip(gen_grads, self.generator.trainable_variables))

    def generate_data(self, n_samples):
        noise = tf.random.normal([n_samples, self.latent_dim])
        return self.generator(noise, training=False).numpy()

def augment_data(original_data):
    gan = WGAN_GP(latent_dim=100, feature_dim=original_data.shape[1])
    gan.train(original_data.values, epochs=100)
    synthetic_data = gan.generate_data(1000)
    return np.vstack([original_data.values, synthetic_data])

def apply_pca(data, n_components=0.95):
    pca = PCA(n_components=n_components)
    reduced_data = pca.fit_transform(data)
    return reduced_data, pca

def wavelet_denoise(signal, wavelet='db8', level=1):
    coeffs = pywt.wavedec(signal, wavelet, level=level)
    threshold = np.std(coeffs[-1]) * np.sqrt(2 * np.log(len(signal)))
    coeffs[1:] = [pywt.threshold(i, threshold, mode='soft') for i in coeffs[1:]]
    return pywt.waverec(coeffs, wavelet)