import os
import sys
import pickle
import tarfile
import urllib.request
import numpy as np

# CIFAR-10 dataset utilities: download, extract, preprocess, and load.

_HERE    = os.path.dirname(os.path.abspath(__file__))
_DATA_DIR = os.path.join(_HERE, 'cifar10')
_TAR_URL  = 'https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz'
_TAR_PATH = os.path.join(_DATA_DIR, 'cifar-10-python.tar.gz')
_EXTRACT_DIR = os.path.join(_DATA_DIR, 'cifar-10-batches-py')


def _reporthook(count, block_size, total_size):
    pct = count * block_size * 100 // total_size
    bar = '#' * (pct // 2) + '-' * (50 - pct // 2)
    sys.stdout.write(f'\r  [{bar}] {pct:3d}%')
    sys.stdout.flush()


def _download():
    os.makedirs(_DATA_DIR, exist_ok=True)
    if not os.path.exists(_TAR_PATH):
        print(f'Downloading CIFAR-10 (~163 MB) from:\n  {_TAR_URL}')
        urllib.request.urlretrieve(_TAR_URL, _TAR_PATH, reporthook=_reporthook)
        print()
    else:
        print(f'  Archive already present: {_TAR_PATH}')


def _extract():
    if not os.path.exists(_EXTRACT_DIR):
        print('Extracting...')
        with tarfile.open(_TAR_PATH, 'r:gz') as tar:
            tar.extractall(_DATA_DIR)
        print('  Done.')
    else:
        print(f'  Already extracted: {_EXTRACT_DIR}')


def _load_batch(path):
    """Load one CIFAR-10 binary batch; return (X float32, y int64)."""
    with open(path, 'rb') as f:
        d = pickle.load(f, encoding='bytes')
    X = d[b'data'].astype(np.float32) / 255.0 
    y = np.array(d[b'labels'], dtype=np.int64)
    return X, y



def _load_raw():
    """Return raw train / test arrays; shape (50000/10000, 3072)."""
    # Training: 5 batches
    X_list, y_list = [], []
    for i in range(1, 6):
        path = os.path.join(_EXTRACT_DIR, f'data_batch_{i}')
        Xi, yi = _load_batch(path)
        X_list.append(Xi); y_list.append(yi)
    X_train = np.vstack(X_list)
    y_train = np.concatenate(y_list)

    # Test batch
    X_test, y_test = _load_batch(os.path.join(_EXTRACT_DIR, 'test_batch'))

    return X_train, X_test, y_train, y_test


def _class_names():
    meta_path = os.path.join(_EXTRACT_DIR, 'batches.meta')
    with open(meta_path, 'rb') as f:
        meta = pickle.load(f, encoding='bytes')
    return [name.decode() for name in meta[b'label_names']]


def _standardise(X_train, X_test):
    from sklearn.preprocessing import StandardScaler
    sc = StandardScaler()
    return sc.fit_transform(X_train), sc.transform(X_test), sc


def _pca_reduce(X_train, X_test, n_components, random_state=42):
    from sklearn.decomposition import PCA
    print(f'  Fitting PCA (n_components={n_components})...')
    pca = PCA(n_components=n_components, random_state=random_state)
    X_tr = pca.fit_transform(X_train)
    X_te = pca.transform(X_test)
    explained = pca.explained_variance_ratio_.sum()
    print(f'  Explained variance: {explained:.1%}')
    return X_tr, X_te, pca


def _npz_path(pca_components):
    if pca_components is None:
        return os.path.join(_DATA_DIR, 'cifar10_raw.npz')
    return os.path.join(_DATA_DIR, f'cifar10_pca{pca_components}.npz')


def _save_npz(X_train, X_test, y_train, y_test, pca_components):
    path = _npz_path(pca_components)
    np.savez_compressed(path,
                        X_train=X_train, X_test=X_test,
                        y_train=y_train, y_test=y_test)
    print(f'  Saved: {path}')
    return path



def prepare(pca_components=None, force=False):

    out_path = _npz_path(pca_components)
    if os.path.exists(out_path) and not force:
        print(f'  NPZ already exists: {out_path}')
        return out_path

    _download()
    _extract()

    print('Loading raw batches...')
    X_train_raw, X_test_raw, y_train, y_test = _load_raw()
    print(f'  Train: {X_train_raw.shape}  Test: {X_test_raw.shape}')

    print('Standardising...')
    X_train, X_test, _ = _standardise(X_train_raw, X_test_raw)

    if pca_components is not None:
        X_train, X_test, _ = _pca_reduce(X_train, X_test, pca_components)

    return _save_npz(X_train, X_test, y_train, y_test, pca_components)


def load_cifar10(pca_components=None, subsample_train=None, random_state=42):

    path = _npz_path(pca_components)
    if not os.path.exists(path):
        prepare(pca_components=pca_components)

    data    = np.load(path)
    X_train = data['X_train'].astype(np.float64)
    X_test  = data['X_test'].astype(np.float64)
    y_train = data['y_train']
    y_test  = data['y_test']

    if subsample_train is not None and subsample_train < len(y_train):
        rng  = np.random.default_rng(random_state)
        idx  = rng.choice(len(y_train), subsample_train, replace=False)
        idx  = np.sort(idx)
        X_train = X_train[idx]
        y_train = y_train[idx]

    return X_train, X_test, y_train, y_test



if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Download & preprocess CIFAR-10')
    parser.add_argument('--pca', type=int, default=None,
                        help='PCA components (default: save raw 3072-dim)')
    parser.add_argument('--force', action='store_true',
                        help='Re-download even if files already exist')
    args = parser.parse_args()

    prepare(pca_components=args.pca, force=args.force)

    if args.pca is None:
        print('\nAlso preparing PCA-256 version...')
        prepare(pca_components=256)

    print('\nClass names:', _class_names())
    print('Done.')
