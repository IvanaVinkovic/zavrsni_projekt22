from flask import Flask, render_template, request
import plotly.graph_objects as go
import numpy as np
import os
import matplotlib as plt 
from sklearn.decomposition import PCA

app = Flask(__name__)

def generate_spheres(n):
    phi = np.random.uniform(0, 2 * np.pi, n)
    theta = np.random.uniform(0, np.pi, n)
    x = np.sin(theta) * np.cos(phi)
    y = np.sin(theta) * np.sin(phi)
    z = np.cos(theta)
    return np.vstack((x, y, z)).T

def add_noise(data, noise_level=0.1):
    noise = np.random.uniform(-noise_level, noise_level, data.shape)
    return data + noise

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        n = int(request.form['n'])
        K1 = generate_spheres(n)
        K2 = generate_spheres(n) + 2
        K3 = generate_spheres(n) + 4

        # Create the Plotly figure for original data
        fig_original = go.Figure()
        fig_original.add_trace(go.Scatter3d(
            x=K1[:, 0], y=K1[:, 1], z=K1[:, 2],
            mode='markers',
            marker=dict(size=3, color='blue'),
            name='K1'
        ))
        fig_original.add_trace(go.Scatter3d(
            x=K2[:, 0], y=K2[:, 1], z=K2[:, 2],
            mode='markers',
            marker=dict(size=3, color='green'),
            name='K2'
        ))
        fig_original.add_trace(go.Scatter3d(
            x=K3[:, 0], y=K3[:, 1], z=K3[:, 2],
            mode='markers',
            marker=dict(size=3, color='red'),
            name='K3'
        ))
        fig_original.update_layout(
            title='Original 3D Spheres',
            scene=dict(
                xaxis_title='X',
                yaxis_title='Y',
                zaxis_title='Z'
            )
        )
        original_path = 'static/original_plot.html'
        fig_original.write_html(original_path)

        data = np.vstack((K1,K2,K3))
        
        # Add noise
        K1_noisy = add_noise(K1)
        K2_noisy = add_noise(K2)
        K3_noisy = add_noise(K3)
        
        data_noisy = np.vstack((K1_noisy,K2_noisy,K3_noisy))

        # Create the Plotly figure for noisy data
        fig_noisy = go.Figure()
        fig_noisy.add_trace(go.Scatter3d(
            x=K1_noisy[:, 0], y=K1_noisy[:, 1], z=K1_noisy[:, 2],
            mode='markers',
            marker=dict(size=3, color='blue'),
            name='K1 (Noisy)'
        ))
        fig_noisy.add_trace(go.Scatter3d(
            x=K2_noisy[:, 0], y=K2_noisy[:, 1], z=K2_noisy[:, 2],
            mode='markers',
            marker=dict(size=3, color='green'),
            name='K2 (Noisy)'
        ))
        fig_noisy.add_trace(go.Scatter3d(
            x=K3_noisy[:, 0], y=K3_noisy[:, 1], z=K3_noisy[:, 2],
            mode='markers',
            marker=dict(size=3, color='red'),
            name='K3 (Noisy)'
        ))
        fig_noisy.update_layout(
            title='Noisy 3D Spheres',
            scene=dict(
                xaxis_title='X',
                yaxis_title='Y',
                zaxis_title='Z'
            )
        )
        noisy_path = 'static/noisy_plot.html'
        fig_noisy.write_html(noisy_path)
        
        # PCA redukcija
        pca = PCA(n_components = 3)
        data_pca = pca.fit_transform(data_noisy)
        
        # PCA Plotly figure for noisy data to original data
        fig_pca = go.Figure()
        fig_pca.add_trace(go.Scatter3d(
            x = data_pca[:,0], y = data_pca[:,1], z = data_pca[:,2],
            mode = 'markers',
            marker = dict(size=3, color = 'green'),
            name = 'PCA reducing'
        ))
        
        fig_pca.update_layout(
            title = 'PCA reduced dimension 3D Spheres',
            scene = dict(
                xaxis_title ='X',
                yaxis_title ='Y',
                zaxis_title ='Z',
            )
        )
        
        pca_path = 'static/pca_plot.html'
        fig_pca.write_html(pca_path)
        
        # SVD reduction and plot
        U, S, VT = np.linalg.svd(data_noisy)
        data_svd = U[:, :3] @ np.diag(S[:3])
        
        fig_svd = go.Figure()
        fig_svd.add_trace(go.Scatter3d(
            x = data_svd[:, 0], y = data_svd[:, 0], z = data_svd[:, 2],
            mode = 'markers',
            marker = dict(size = 3, color = 'red'),
            name = 'SVD reducing'
        ))
        
        fig_svd.update_layout(
            title = 'SVD reduced dimension 3D spheres',
            scene = dict(
                xaxis_title = 'X',
                yaxis_title = 'Y',
                zaxis_title = 'Z'
            )
        )
        
        svd_path = 'static/svd_plot.html'
        fig_svd.write_html(svd_path)
        
        return render_template('index.html', original_plot=original_path, noisy_plot=noisy_path)

    return render_template('index.html', original_plot=None, noisy_plot=None)

if __name__ == '__main__':
    if not os.path.exists('static'):
        os.makedirs('static')
    app.run(debug=True)
