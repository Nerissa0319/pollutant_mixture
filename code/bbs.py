import numpy as np

def block_bootstrap(data, block_size, num_resamples):
    n = len(data)
    num_blks = n - block_size + 1
    resamples = []
    blks = []
    for i in range(num_blks):
        start_idx = i
        end_idx = start_idx + block_size
        blks.append(data[start_idx:end_idx])
    for _ in range(num_resamples):
        resampled_series = []
        resampled_indices = np.random.choice(num_blks, size=n // block_size, replace=True)

        for j in resampled_indices:
            resampled_series.extend(blks[j])
        resamples.append(np.array(resampled_series))
    return resamples