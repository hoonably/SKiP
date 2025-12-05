# datasets/download.py

import os
import numpy as np
import pandas as pd
from sklearn.datasets import load_wine, load_breast_cancer, load_iris
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


def save_npz(path, X, y):
    dirpath = os.path.dirname(path)
    if dirpath:
        os.makedirs(dirpath, exist_ok=True)
    np.savez(path, X_train=X, y_train=y)
    print(f"\nSaved: {path}")

    with np.load(path) as tmp:
        print("Check:", tmp["X_train"].shape, tmp["y_train"].shape, " | unique labels:", len(np.unique(tmp["y_train"])))


def apply_pca(X, n_components=0.95):
    """Apply PCA to reduce dimensionality to n_components"""
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    pca = PCA(n_components=n_components)
    X_pca = pca.fit_transform(X_scaled)
    print(f"\nApplied PCA: reduced to {pca.n_components_} dimensions.")
    print(f"PCA explained variance ratio: {pca.explained_variance_ratio_}")
    return X_pca


def download_wine(save_dir):
    wine = load_wine()
    X, y = wine.data, wine.target
    path = os.path.join(save_dir, "wine", "wine.npz")
    save_npz(path, X, y)
    
    # PCA version
    X_pca = apply_pca(X)
    path_pca = os.path.join(save_dir, "wine_pca", "wine_pca.npz")
    save_npz(path_pca, X_pca, y)


def download_breast_cancer(save_dir):
    data = load_breast_cancer()
    X, y = data.data, data.target
    path = os.path.join(save_dir, "breast_cancer", "breast_cancer.npz")
    save_npz(path, X, y)
    
    # PCA version
    X_pca = apply_pca(X)
    path_pca = os.path.join(save_dir, "breast_cancer_pca", "breast_cancer_pca.npz")
    save_npz(path_pca, X_pca, y)


def download_iris(save_dir):
    iris = load_iris()
    X = iris.data
    y = iris.target
    path = os.path.join(save_dir, "iris", "iris.npz")
    save_npz(path, X, y)
    
    # PCA version
    X_pca = apply_pca(X)
    path_pca = os.path.join(save_dir, "iris_pca", "iris_pca.npz")
    save_npz(path_pca, X_pca, y)


def download_titanic(save_dir):
    # Use local titanic.zip file
    import tempfile
    import zipfile
    
    # Path to local titanic.zip in the same directory as this script
    script_dir = os.path.dirname(os.path.abspath(__file__))
    zip_path = os.path.join(script_dir, "titanic.zip")
    
    if not os.path.exists(zip_path):
        raise FileNotFoundError(f"titanic.zip not found at {zip_path}")
    
    with tempfile.TemporaryDirectory() as tmpdir:
        # Unzip the local file
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(tmpdir)
        
        # Load train.csv
        train_path = os.path.join(tmpdir, "train.csv")
        df = pd.read_csv(train_path)
    
    # === Preprocess the data ===
    # Select target variable (Survived)
    y = df['Survived'].values
    
    # Select features and handle missing values
    # Use numerical features: Pclass, Age, SibSp, Parch, Fare
    # Use categorical features: Sex, Embarked
    # Unused features: PassengerId, Name, Ticket, Cabin (not useful or too many missing)
    features = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked']
    df_features = df[features].copy()
    
    # Fill missing values
    df_features['Age'] = df_features['Age'].fillna(df_features['Age'].median())
    df_features['Fare'] = df_features['Fare'].fillna(df_features['Fare'].median())
    df_features['Embarked'] = df_features['Embarked'].fillna(df_features['Embarked'].mode()[0])
    
    # Encode categorical variables (One-hot encoding)
    df_features = pd.get_dummies(df_features, columns=['Sex', 'Embarked'], drop_first=True)
    
    # Convert to numpy array
    X = df_features.values.astype(float)
    
    # Remove rows with any remaining NaN values
    valid_idx = ~np.isnan(X).any(axis=1) & ~np.isnan(y)
    X = X[valid_idx]
    y = y[valid_idx]
    
    path = os.path.join(save_dir, "titanic", "titanic.npz")
    save_npz(path, X, y)
    
    # PCA version
    X_pca = apply_pca(X)
    path_pca = os.path.join(save_dir, "titanic_pca", "titanic_pca.npz")
    save_npz(path_pca, X_pca, y)


def main():
    save_dir = "."
    os.makedirs(save_dir, exist_ok=True)

    download_wine(save_dir)
    download_breast_cancer(save_dir)
    download_iris(save_dir)
    download_titanic(save_dir)


if __name__ == "__main__":
    main()
