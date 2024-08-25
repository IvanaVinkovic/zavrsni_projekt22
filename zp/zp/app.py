import numpy as np
import plotly.graph_objects as go
import os
from flask import Flask, render_template, request
from sklearn.decomposition import PCA
import numpy as np
import tensorflow as tf
import torch 
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt 

app = Flask(__name__)

def generate_spheres(n):
    phi = np.random.uniform(0, 2 * np.pi, n)
    theta = np.random.uniform(0, np.pi, n)
    x = np.sin(theta) * np.cos(phi)
    y = np.sin(theta) * np.sin(phi)
    z = np.cos(theta)
    return np.vstack((x, y, z)).T

def add_noise(data, noise_points):
    noise = np.random.uniform(-10, 15, (noise_points, data.shape[1]))
    return np.vstack((data, noise))

def permute_coordinates(data):
    permuted_data = np.copy(data)
    np.random.shuffle(permuted_data.T)  
    return permuted_data

# Autoencoder Model
def build_autoencoder(input_dim, encoding_dim):
    # Koder
    input_layer = tf.keras.layers.Input(shape=(input_dim,))
    encoder = tf.keras.layers.Dense(64, activation="relu")(input_layer)
    encoder = tf.keras.layers.Dense(32, activation="relu")(encoder)
    latent_space = tf.keras.layers.Dense(encoding_dim, activation="relu")(encoder)  # Latentni prostor

    # Dekoder
    decoder = tf.keras.layers.Dense(32, activation="relu")(latent_space)
    decoder = tf.keras.layers.Dense(64, activation="relu")(decoder)
    output_layer = tf.keras.layers.Dense(input_dim, activation="sigmoid")(decoder)  # Rekonstrukcija

    # Model autoencodera
    autoencoder = tf.keras.models.Model(inputs=input_layer, outputs=output_layer)
    autoencoder.compile(optimizer="adam", loss="mse")
    
    # Model kodera za smanjenje dimenzija
    encoder_model = tf.keras.models.Model(inputs=input_layer, outputs=latent_space)
    
    return autoencoder, encoder_model

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        n = int(request.form['n'])
        
        # Generiranje i dodavanje podataka
        K1 = generate_spheres(n)
        K2 = generate_spheres(n) + 5
        K3 = generate_spheres(n) + 10

        # Dodavanje šuma
        data = np.vstack((K1, K2, K3))
        data_with_noise = add_noise(data, 200)
        
        noise = np.random.uniform(0, 1, (data_with_noise.shape[0], 7))
        data_noisy = np.hstack((data_with_noise,noise))
        data_permuted = permute_coordinates(data_noisy)

        # Vizualizacija originalnih podataka sa šumom
        fig_original = go.Figure()
        fig_original.add_trace(go.Scatter3d(
            x=K1[:, 0], y=K1[:, 1], z=K1[:, 2],
            mode='markers',
            marker=dict(size=5, color='red'),
            name='Kugla 1'
        ))
        fig_original.add_trace(go.Scatter3d(
            x=K2[:, 0], y=K2[:, 1], z=K2[:, 2],
            mode='markers',
            marker=dict(size=5, color='green'),
            name='Kugla 2'
        ))
        fig_original.add_trace(go.Scatter3d(
            x=K3[:, 0], y=K3[:, 1], z=K3[:, 2],
            mode='markers',
            marker=dict(size=5, color='blue'),
            name='Kugla 3'
        ))
        # Dodavanje šuma na graf originalnih podataka
        fig_original.add_trace(go.Scatter3d(
            x=data_with_noise[len(data):, 0], y=data_with_noise[len(data):, 1], z=data_with_noise[len(data):, 2],
            mode='markers',
            marker=dict(size=2, color='gray', symbol='x'),
            name='Šum'
        ))
        fig_original.update_layout(
            title='Originalni podaci sa šumom',
            scene=dict(
                xaxis_title='X',
                yaxis_title='Y',
                zaxis_title='Z'
            )
        )
        original_path = 'static/original_plot.html'
        fig_original.write_html(original_path)
        
        #PCA
        pca = PCA(n_components = 3)
        data_pca = pca.fit_transform(data_permuted)
        
        total_points = data_pca.shape[0]
        n_kugli = n * 3
        
        fig_pca = go.Figure()
        fig_pca.add_trace(go.Scatter3d(
            x=data_pca[:n, 0],
            y=data_pca[:n, 1],
            z=data_pca[:n, 2],
            mode='markers',
            marker=dict(size=3, color='red'),
            name='K1 PCA'
        ))
        
        fig_pca.add_trace(go.Scatter3d(
            x=data_pca[n:2*n, 0],
            y=data_pca[n:2*n, 1],
            z=data_pca[n:2*n, 2],
            mode='markers',
            marker=dict(size=3, color='green'),
            name='K2 PCA'
        ))      
        
        fig_pca.add_trace(go.Scatter3d(
            x=data_pca[2*n:3*n, 0],
            y=data_pca[2*n:3*n, 1],
            z=data_pca[2*n:3*n, 2],
            mode='markers',
            marker=dict(size=3, color='blue'),
            name='K3 PCA'
        ))
        
        # noise       
        fig_pca.add_trace(go.Scatter3d(
            x=data_pca[n_kugli:, 0], 
            y=data_pca[n_kugli:, 1],
            z=data_pca[n_kugli:, 2],
            mode='markers',
            marker=dict(size=2, color='gray', symbol='x'),
            name='Šum'
        ))
        
        fig_pca.update_layout(
            title='PCA podaci sa šumom',
            scene=dict(
                xaxis_title='X',
                yaxis_title='Y',
                zaxis_title='Z'
            )
        )        
        
        pca_path = 'static/pca_plot.html'
        fig_pca.write_html(pca_path)
        
        #SVD
        U ,S , VT = np.linalg.svd(data_permuted)
        data_svd = U[:, :3] @ np.diag(S[:3])
        
        fig_svd = go.Figure()
        fig_svd.add_trace(go.Scatter3d(
            x=data_svd[:n, 0],
            y=data_svd[:n, 1],
            z=data_svd[:n, 2],
            mode='markers',
            marker=dict(size=3, color='red'),
            name='K1 SVD'
        ))
        
        fig_svd.add_trace(go.Scatter3d(
            x=data_svd[n:2*n, 0],
            y=data_svd[n:2*n, 1],
            z=data_svd[n:2*n, 2],
            mode='markers',
            marker=dict(size=3, color='green'),
            name='K2 SVD'
        ))      
        
        fig_svd.add_trace(go.Scatter3d(
            x=data_svd[2*n:3*n, 0],
            y=data_svd[2*n:3*n, 1],
            z=data_svd[2*n:3*n, 2],
            mode='markers',
            marker=dict(size=3, color='blue'),
            name='K3 SVD'
        ))
        
        # noise
        fig_svd.add_trace(go.Scatter3d(
            x=data_svd[n_kugli:, 0], 
            y=data_svd[n_kugli:, 1],
            z=data_svd[n_kugli:, 2],
            mode='markers',
            marker=dict(size=2, color='gray', symbol='x'),
            name='Šum'
        ))
        
        fig_svd.update_layout(
            title = 'SVD sa šumom',
            scene = dict(
                xaxis_title='X',
                yaxis_title='Y',
                zaxis_title='Z'                
            )
        )
        
        svd_path = 'static/svd_plot.html'
        fig_svd.write_html(svd_path)
        
        # mMDS
        class MDSNet(tf.keras.Model):
            def __init__(self, input_dim, hidden_dims, output_dim):
                super(MDSNet, self).__init__()
                self.hidden_layers = []
                
                for hidden_dim in hidden_dims:
                    self.hidden_layers.append(
                        tf.keras.layers.Dense(hidden_dim, activation='elu', kernel_regularizer=tf.keras.regularizers.l2(1e-5))
                    )
                    self.hidden_layers.append(tf.keras.layers.Dropout(0.2))
                
                self.output_layer = tf.keras.layers.Dense(output_dim, activation=None)
            
            def call(self, x):
                for layer in self.hidden_layers:
                    x = layer(x)
                
                return self.output_layer(x)
        
        def pairwise_distances(x):
            dot_product = tf.matmul(x, tf.transpose(x))
            square_norm = tf.linalg.diag_part(dot_product)
            distances = tf.expand_dims(square_norm, 0) - 2.0 * dot_product + tf.expand_dims(square_norm, 1)
            distances = tf.sqrt(tf.maximum(distances, 1e-9))
            return distances

        input_dim = data_noisy.shape[1]
        hidden_dims = [128, 64, 32]
        output_dim = 3

        model_mmds = MDSNet(input_dim, hidden_dims, output_dim)
        optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)

        @tf.function
        def train_step(batch_data):
            with tf.GradientTape() as tape:
                output = model_mmds(batch_data)
                distance_orig = pairwise_distances(batch_data)
                distance_proj = pairwise_distances(output)
                loss = tf.reduce_mean(tf.square(distance_proj - distance_orig))
            gradients = tape.gradient(loss, model_mmds.trainable_variables)
            optimizer.apply_gradients(zip(gradients, model_mmds.trainable_variables))
            return loss

        labels = np.concatenate([
        np.zeros(n),  
        np.ones(n),   
        np.full(n, 2) 
        ])

        additional_labels = np.full(200, -1) 
        labels = np.concatenate([labels, additional_labels])

        epochs = 3000
        batch_size = 32
        losses = []

        data_noisy_tensor = tf.convert_to_tensor(data_noisy, dtype=tf.float32)

        for epoch in range(epochs):
            permutation = np.random.permutation(data_noisy.shape[0])
            for i in range(0, data_noisy.shape[0], batch_size):
                indices = permutation[i:i+batch_size]
                batch_data = tf.gather(data_noisy_tensor, indices)
                loss = train_step(batch_data)
            losses.append(loss.numpy())
            if (epoch + 1) % 500 == 0:
                print(f'Epoch [{epoch + 1}/{epochs}], Loss: {loss:.4f}')

        output_data = model_mmds(data_noisy_tensor).numpy()

        fig_mMDS = go.Figure()
        for label, color, name in zip([0, 1, 2], ['red', 'green', 'blue'], ['K1', 'K2', 'K3']):
            fig_mMDS.add_trace(go.Scatter3d(
                x=output_data[labels == label, 0],
                y=output_data[labels == label, 1],
                z=output_data[labels == label, 2],
                mode='markers',
                marker=dict(size=5, color=color),
                name=f'{name} mMDS'
            ))

        # noise
        if np.any(labels == -1):
            fig_mMDS.add_trace(go.Scatter3d(
                x=output_data[labels == -1, 0],
                y=output_data[labels == -1, 1],
                z=output_data[labels == -1, 2],
                mode='markers',
                marker=dict(size=3, color='grey', symbol='x'),
                name='Šum'
            ))

        fig_mMDS.update_layout(
            title='mMDS sa šumom',
            scene=dict(
                xaxis_title='X',
                yaxis_title='Y',
                zaxis_title='Z'
            )
        )
        
        mMDS_path = 'static/mMDS_plot.html'
        fig_mMDS.write_html(mMDS_path)
        
        #Autoencoder
        encoding_dim = int(request.form['encoding_dim'])
        input_dim = data_noisy.shape[1]
        autoencoder, encoder_model = build_autoencoder(input_dim, encoding_dim)
        autoencoder.fit(data_noisy, data_noisy, epochs=100, batch_size=32, verbose=0)
        autoencoder_path = 'static/autoencoder_plot.html'
        
        # Smanjenje dimenzija i rekonstrukcija
        reduced_data = encoder_model.predict(data_noisy)
        reconstructed_data = autoencoder.predict(data_noisy)
        
        # Interaktivna vizualizacija latentnog prostora
        fig_reduced = go.Figure()
        fig_reduced.add_trace(go.Scatter3d(
            x=reduced_data[:n, 0], y=reduced_data[:n, 1], z=reduced_data[:n, 2],
            mode='markers',
            marker=dict(size=5, color='red'),
            name='Kugla 1 - Latentni prostor'
        ))
        fig_reduced.add_trace(go.Scatter3d(
            x=reduced_data[n:2*n, 0], y=reduced_data[n:2*n, 1], z=reduced_data[n:2*n, 2],
            mode='markers',
            marker=dict(size=5, color='green'),
            name='Kugla 2 - Latentni prostor'
        ))
        fig_reduced.add_trace(go.Scatter3d(
            x=reduced_data[2*n:3*n, 0], y=reduced_data[2*n:3*n, 1], z=reduced_data[2*n:3*n, 2],
            mode='markers',
            marker=dict(size=5, color='blue'),
            name='Kugla 3 - Latentni prostor'
        ))

        reduced_path = 'static/reduced_plot.html'
        fig_reduced.write_html(reduced_path)

        # Interaktivna vizualizacija rekonstruiranih podataka
        """
        fig_reconstructed = go.Figure()
        fig_reconstructed.add_trace(go.Scatter3d(
            x=reconstructed_data[:n, 0], y=reconstructed_data[:n, 1], z=reconstructed_data[:n, 2],
            mode='markers',
            marker=dict(size=5, color='red'),
            name='Kugla 1 - Rekonstrukcija'
        ))
        fig_reconstructed.add_trace(go.Scatter3d(
            x=reconstructed_data[n:2*n, 0], y=reconstructed_data[n:2*n, 1], z=reconstructed_data[n:2*n, 2],
            mode='markers',
            marker=dict(size=5, color='green'),
            name='Kugla 2 - Rekonstrukcija'
        ))
        fig_reconstructed.add_trace(go.Scatter3d(
            x=reconstructed_data[2*n:3*n, 0], y=reconstructed_data[2*n:3*n, 1], z=reconstructed_data[2*n:3*n, 2],
            mode='markers',
            marker=dict(size=5, color='blue'),
            name='Kugla 3 - Rekonstrukcija'
        ))

        reconstructed_path = 'static/reconstructed_plot.html'
        fig_reconstructed.write_html(reconstructed_path)
        
        fig_reconstructed.write_html(autoencoder_path)"""

        return render_template('index.html', 
                               original_plot=original_path, 
                               pca_plot = pca_path, svd_plot = svd_path, mMDS_plot = mMDS_path, reduced_plot=reduced_path)

    return render_template('index.html', original_plot=None, pca_plot = None, svd_plot = None, mMDS_plot = None, autoencoder_plot = None, reduced_plot=None, reconstructed_plot=None)

if __name__ == '__main__':
    if not os.path.exists('static'):
        os.makedirs('static')
    app.run(debug=True)
