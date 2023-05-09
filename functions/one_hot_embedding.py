
def embed_label(x, y):
    """add digit representation to input pixels"""
    x_ = x.clone()
    x_[:, :10] *= 0.0
    x_[range(x.shape[0]), y] = x.max()
    return x_
