def make_channels(num_layers, reverse=False):
    channel_length = num_layers + 1
    _base = [2**i for i in range(4, 10)]
    if reverse:
        _base = _base[::-1]
    residual = channel_length - len(_base)
    if residual < 0:
        return _base[abs(int(residual)):]
    if residual == 0:
        return _base
    channels = list()
    for c in _base:
        if residual > 0:
            channels += [c] * 2
            residual -= 1
        else:
            channels += [c]
    return channels