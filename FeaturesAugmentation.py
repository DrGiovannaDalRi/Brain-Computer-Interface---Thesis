###FEATURES AUGMENTATION THROUGH WGAN-GP ALGORITHM
"""
Pipeline:
1. Load extracted EEG features
2. Normalize features using StandardScaler
3. Train a condition-specific WGAN-GP
4. Generate synthetic feature vectors
5. Apply biological plausibility constraints
6. Validate generated samples
7. Save augmented datasets and trained generators

Conditions:
- Right-hand motor imagery
- Left-hand motor imagery
- Idle state
"""

##IMPORT LIBRARIES

#Operating system utilities
import os

#Data handling and numerical computations
import pandas as pd
import numpy as np

#Visualization of training results and validation analyses
import matplotlib.pyplot as plt
import seaborn as sns

#Statistical validation of synthetic feature distribution
from scipy import stats

#Deep learning framework used of WGAN-GP implementation
import tensorflow as tf
from tensorflow.keras import layers, models

#Feature normalization before GAN training
from sklearn.preprocessing import StandardScaler

#Dimensionality reduction for PCA-based validation
from sklearn.decomposition import PCA


#Base class for EEG condition processing
class EEGConditionProcessor:

    def __init__(self, condition_name, file_path):
        self.condition = condition_name
        self.file_path = file_path
        self.scaler = None
        self.original_data = None
        self.normalized_data = None
        self.generator = None

    def load_and_normalize(self):
        """Load and normalize EEG feature file"""
        self.original_data = pd.read_excel(self.file_path)
        self.scaler = StandardScaler()
        self.normalized_data = self.scaler.fit_transform(self.original_data.values)
        return self

#WGAN-GP model for specific EEG condition
class EEGConditionGAN(EEGConditionProcessor):

    def __init__(self, condition_name, file_path, latent_dim=100, n_critic=2, gp_weight=8):
        super().__init__(condition_name, file_path)
        self.latent_dim = latent_dim
        self.n_critic = n_critic
        self.gp_weight = gp_weight
        self.critic = None
        self.optimal_params = {
            'Right': {'latent_dim': 120, 'gp_weight': 10, 'n_critic': 3},
            'Left': {'latent_dim': 150, 'gp_weight': 15, 'n_critic': 5},
            'Idle': {'latent_dim': 80, 'gp_weight': 8, 'n_critic': 2}
        }

    #Generator network
    def build_generator(self):
        model = models.Sequential([
            layers.Input(shape=(self.latent_dim,)),
            layers.Dense(256, kernel_initializer='he_normal'),
            layers.LeakyReLU(negative_slope=0.2),
            layers.BatchNormalization(momentum=0.9),
            layers.Dropout(0.25),
            layers.Dense(384, kernel_initializer='he_normal'),
            layers.LeakyReLU(negative_slope=0.2),
            layers.BatchNormalization(momentum=0.9),
            layers.Dense(self.original_data.shape[1], activation='linear')
        ])
        return model

    #Critic network
    def build_critic(self):
        """Critic network architecture"""
        model = models.Sequential([
            layers.Input(shape=(self.original_data.shape[1],)),
            layers.Dense(384, kernel_initializer='he_normal'),
            layers.LeakyReLU(negative_slope=0.2),
            layers.Dropout(0.3),
            layers.Dense(192, kernel_initializer='he_normal'),
            layers.LeakyReLU(negative_slope=0.2),
            layers.Dense(1)
        ])
        return model

    #Compute gradient penalty GP for WGAN
    def gradient_penalty(self, real_samples, fake_samples):

        batch_size = real_samples.shape[0]
        epsilon = tf.random.uniform([batch_size, 1], 0.0, 1.0)
        interpolated = epsilon * real_samples + (1 - epsilon) * fake_samples

        with tf.GradientTape() as tape:
            tape.watch(interpolated)
            validity = self.critic(interpolated)

        gradients = tape.gradient(validity, interpolated)
        gradients = tf.reshape(gradients, [gradients.shape[0], -1])
        penalty = tf.reduce_mean((tf.norm(gradients, axis=1) - 1) ** 2)
        return penalty

    #Train the GAN on EEG features
    def train(self, epochs=6000, batch_size=12):
        self.generator = self.build_generator()
        self.critic = self.build_critic()

        # Condition-specific learning rates
        lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
            initial_learning_rate=0.0002,
            decay_steps=1000,
            decay_rate=0.9)

        g_optimizer = tf.keras.optimizers.Adam(lr_schedule, beta_1=0.5)
        c_optimizer = tf.keras.optimizers.Adam(0.00002, beta_1=0.5)

        for epoch in range(epochs):
            # Train critic
            for _ in range(self.n_critic):
                idx = np.random.randint(0, self.normalized_data.shape[0], batch_size)
                real = self.normalized_data[idx]
                noise = tf.random.normal([batch_size, self.latent_dim])
                fake = self.generator(noise, training=True)

                with tf.GradientTape() as tape:
                    real_validity = self.critic(real, training=True)
                    fake_validity = self.critic(fake, training=True)
                    gp = self.gradient_penalty(real, fake)
                    c_loss = tf.reduce_mean(fake_validity) - tf.reduce_mean(real_validity) + self.gp_weight * gp

                c_grads = tape.gradient(c_loss, self.critic.trainable_variables)
                c_optimizer.apply_gradients(zip(c_grads, self.critic.trainable_variables))

            # Train generator
            noise = tf.random.normal([batch_size, self.latent_dim])
            with tf.GradientTape() as tape:
                fake = self.generator(noise, training=True)
                g_loss = -tf.reduce_mean(self.critic(fake, training=True))

            g_grads = tape.gradient(g_loss, self.generator.trainable_variables)
            g_optimizer.apply_gradients(zip(g_grads, self.generator.trainable_variables))

            # Enhanced monitoring
            if epoch % 100 == 0:
                # Generate sample features for monitoring
                samples = self.generate_synthetic(n_samples=5)
                psd_idx = [i for i, col in enumerate(self.original_data.columns) if 'PSD' in col][0]
                print(
                    f"Epoch {epoch} | C Loss: {c_loss:.4f} | G Loss: {g_loss:.4f} | Sample PSD: {samples[0, psd_idx]:.3f}")

                # Early stopping condition
                if epoch > 1000 and abs(float(c_loss)) < 0.8 and float(g_loss) < -0.8:
                    print(f"Early stopping at epoch {epoch}")
                    break

        return self

    #Biological constraints to generated features
    def clean_features(self, features):
        clean = features.copy()
        for i, col in enumerate(self.original_data.columns):
            if 'PSD' in col:
                clean[:, i] = np.clip(features[:, i], 0.1, 0.6)

                # Special scaling for Left condition PSD
                if self.condition == 'Left':
                    clean[:, i] *= 0.9
            elif 'amplitude' in col:
                clean[:, i] = np.clip(features[:, i], -15, 15)
            elif 'kurtosis' in col:
                clean[:, i] = np.clip(features[:, i], 2.0, 4.0)
        return clean

    #Generate and clean synthetic features
    def generate_synthetic(self, n_samples=None):

        if n_samples is None:
            n_samples = len(self.original_data) * 2  # Default 2x augmentation

        noise = tf.random.normal([n_samples, self.latent_dim])
        synthetic = self.scaler.inverse_transform(self.generator.predict(noise))
        return self.clean_features(synthetic)

    #Enhanced validation with correlation analysis
    def validate(self, synthetic_features):

        synthetic_df = pd.DataFrame(synthetic_features, columns=self.original_data.columns)

        # Use same number of samples for validation
        n_samples = min(len(self.original_data), len(synthetic_df))
        real_samples = self.original_data.sample(n_samples)
        syn_samples = synthetic_df.sample(n_samples)

        # Feature validation
        for feature in ['PSD-10-11', 'mean_amplitude', 'kurtosis']:
            if feature in self.original_data.columns:
                self._feature_report(
                    real_samples[feature].values,
                    syn_samples[feature].values,
                    feature
                )

        # PCA visualization
        self._pca_validation(real_samples.values, syn_samples.values)

        # Feature correlation analysis with matching sample sizes
        self._correlation_analysis(real_samples, syn_samples)

    #Feature comparison plots
    def _feature_report(self, real, synthetic, feature_name):

        plt.figure(figsize=(12, 5))

        # Distribution plot
        plt.subplot(1, 2, 1)
        plt.hist(real, bins=30, alpha=0.5, label='Real', density=True)
        plt.hist(synthetic, bins=30, alpha=0.5, label='Synthetic', density=True)
        plt.title(f"{self.condition} - {feature_name} Distribution")
        plt.xlabel(feature_name)
        plt.ylabel("Density")
        plt.legend()

        # Q-Q plot
        plt.subplot(1, 2, 2)
        stats.probplot(real, dist="norm", plot=plt)
        stats.probplot(synthetic, dist="norm", plot=plt)
        plt.title(f"{self.condition} - {feature_name} Q-Q Plot")
        plt.xlabel("Theoretical Quantiles")
        plt.ylabel("Sample Quantiles")

        plt.tight_layout()
        os.makedirs('validation_plots', exist_ok=True)
        plt.savefig(f"validation_plots/{self.condition}_{feature_name}_validation.png", dpi=300)
        plt.close()

        # Statistical tests
        ks_stat, ks_p = stats.ks_2samp(real, synthetic)
        print(f"\n{self.condition} - {feature_name} Validation:")
        print(f"KS Test: D={ks_stat:.3f}, p={ks_p:.3f}")
        print(f"Real: μ={np.mean(real):.3f} ± {np.std(real):.3f}")
        print(f"Synth: μ={np.mean(synthetic):.3f} ± {np.std(synthetic):.3f}")

    #PCA validation
    def _pca_validation(self, real, synthetic):

        pca = PCA(n_components=2)
        combined = np.vstack([real, synthetic])
        pca_results = pca.fit_transform(combined)

        plt.figure(figsize=(10, 8))
        plt.scatter(pca_results[:len(real), 0], pca_results[:len(real), 1],
                    alpha=0.3, label='Real', s=30)
        plt.scatter(pca_results[len(real):, 0], pca_results[len(real):, 1],
                    alpha=0.3, label='Synthetic', s=30)
        plt.title(f"{self.condition} - Feature Space (PCA)")
        plt.xlabel("Principal Component 1")
        plt.ylabel("Principal Component 2")
        plt.legend()
        plt.savefig(f"validation_plots/{self.condition}_pca.png", dpi=300)
        plt.close()

    #Feature correlation analysis with matching sample sizes
    def _correlation_analysis(self, real_samples, syn_samples):

        # Calculate feature-wise correlations
        feature_corr = []
        for col in real_samples.columns:
            r, p = stats.pearsonr(real_samples[col], syn_samples[col])
            feature_corr.append(r)

        # Create correlation dataframe
        corr_df = pd.DataFrame({
            'Feature': real_samples.columns,
            'Pearson_r': feature_corr
        })

        # Plot correlations
        plt.figure(figsize=(10, 6))
        sns.barplot(x='Pearson_r', y='Feature', data=corr_df, palette='viridis')
        plt.title(f"{self.condition} Feature Correlations (Real vs Synthetic)")
        plt.xlim(-1, 1)
        plt.axvline(0, color='black', linestyle='--')
        plt.tight_layout()
        plt.savefig(f"validation_plots/{self.condition}_feature_correlations.png", dpi=300)
        plt.close()

        # Print correlation summary
        print(f"\n{self.condition} Feature Correlations:")
        print(corr_df.sort_values('Pearson_r', ascending=False).to_string(index=False))

    #Save augmented features dataset
    def save_results(self, synthetic_features):

        augmented_df = pd.DataFrame(synthetic_features, columns=self.original_data.columns)

        # Save with exact requested names
        augmented_df.to_excel(f"AugmentationFeatures{self.condition}.xlsx", index=False)
        os.makedirs('augmented_data', exist_ok=True)
        augmented_df.to_excel(f"augmented_data/AugmentationFeatures{self.condition}.xlsx", index=False)

        # Save model
        os.makedirs('saved_models', exist_ok=True)
        self.generator.save(f"saved_models/generator_{self.condition}.keras")


def main():
    # Configuration with condition-specific parameters
    conditions = {
        'Right': EEGConditionGAN('Right', 'featuresRight.xlsx',
                                 latent_dim=120, gp_weight=10, n_critic=3),
        'Left': EEGConditionGAN('Left', 'featuresLeft.xlsx',
                                latent_dim=150, gp_weight=15, n_critic=5),
        'Idle': EEGConditionGAN('Idle', 'featuresIdle.xlsx',
                                latent_dim=80, gp_weight=8)
    }

    # Create output directories
    os.makedirs('augmented_data', exist_ok=True)
    os.makedirs('validation_plots', exist_ok=True)
    os.makedirs('saved_models', exist_ok=True)

    for condition, processor in conditions.items():

        # Load and normalize data
        processor.load_and_normalize()

        # Train GAN
        processor.train()

        # Generate synthetic features
        synthetic_features = processor.generate_synthetic()

        # Save results and validate
        processor.save_results(synthetic_features)
        processor.validate(synthetic_features)

if __name__ == "__main__":
    main()