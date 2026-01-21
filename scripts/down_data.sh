from datasets import load_dataset

# # Login using e.g. `huggingface-cli login` to access this dataset
# ds = load_dataset("ysmao/arkitscenes-spatiallm")
# print(ds)


# # Login using e.g. `huggingface-cli login` to access this dataset
# ds = load_dataset("ysmao/structured3d-spatiallm")
# print(ds)

# we locally download the arkitscenes-spatiallm, structured3d-spatiallm
# used command to Download the dataset repo
# (spatiallm) jinjingxu@G27WS0014-Linux:/mnt/nct-zfs$ cd TCO-All/SharedDatasets/
# (spatiallm) jinjingxu@G27WS0014-Linux:/mnt/nct-zfs/TCO-All/SharedDatasets$ huggingface-cli download ysmao/structured3d-spatiallm --repo-type dataset --local-dir structured3d-spatiallm

# huggingface-cli download ysmao/structured3d-spatiallm \
#     --repo-type dataset \
#     --local-dir structured3d-spatiallm \
#     --local-dir-use-symlinks False