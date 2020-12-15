import librl.replay

# Return a replay buffer that allows recording of arbitray data in .extra
def ProductEpisodeWithExtraLogs(*args, **kwargs):
    return librl.replay.ProductEpisode(*args, enable_extra=True, **kwargs)