from preprocessing import data_handle
import numpy as np

filepath = "/Volumes/Extreme Pro/Chapter1_publication_files/testdata/data/20251012/dphi/200008.hdf5"   # <-- replace this
channel_range = [30000, 90000, 2]

md = data_handle.get_acquisition_parameters(filepath)
selected = data_handle.select_channels(md["dx"], channel_range)
trace, tx, dist = data_handle.load_das_data(filepath, selected, md)

print("metadata:", md)
print("trace shape:", trace.shape)
print("dist range (m):", dist.min(), dist.max())

trace_rs, tx_rs, fs_rs = data_handle.resample_trace_to_target_fs(trace, tx, md["fs"], 256.0)
print("resampled trace shape:", trace_rs.shape)
print("resampled fs:", fs_rs)

seg_defs = data_handle.build_segments_from_dist(dist, channel_range, seg_len_m=20000, min_channels=8)
print("number of segments:", len(seg_defs))
for slc, seg_tuple, label in seg_defs:
    print(" segment:", label, "n_channels:", slc.stop - slc.start)

zmat = data_handle.channel_block_scores(trace_rs, fs_rs, block_length_s=1)
print("z-score matrix shape:", zmat.shape)

fake_bad_idx = np.array([5, 10, 20])
trace_clean = data_handle.suppress_bad_channels(trace_rs, fake_bad_idx, mode="interp")
print("suppressed trace shape:", trace_clean.shape)