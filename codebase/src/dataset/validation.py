def validate_dataset(dataset):
    if dataset.data_real is None or dataset.data_sim is None:
        raise RuntimeError("Dataset is incomplete. Missing real or sim data.")
    if len(dataset.data_real) == 0 or len(dataset.data_sim) == 0:
        raise RuntimeError("Loaded dataset is empty.")
    if len(dataset.data_real) != len(dataset.data_sim):
        raise RuntimeError("Mismatch between number of real and sim samples.")
