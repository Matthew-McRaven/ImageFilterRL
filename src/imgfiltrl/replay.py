import librl.replay


def ProductEpisodeWithExtraLogs(*args, **kwargs):
    return librl.replay.ProductEpisode(*args, enable_extra=True, **kwargs)