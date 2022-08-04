import numpy as np

from src.data import calc_balanced_sample, split

def test_calc_balanced_sample():
    class_counts = [10, 100, 100]
    n_samples = 60
    sample = calc_balanced_sample(class_counts, n_samples)
    assert sample[0] == 10
    assert sample[1] == 25
    assert sample[2] == 25
    class_counts = [100, 100, 100]
    n_samples = 60
    sample = calc_balanced_sample(class_counts, n_samples)
    assert sample[0] == 20
    assert sample[1] == 20
    assert sample[2] == 20

def test_split():
    labels = np.zeros(20)
    labels[:10] = 1
    data_params = dict(
        classes = 2,
        n_per_class_trn = 5,
        n = 20
    )
    split_trn, split_val = split(labels, data_params)

    assert split_trn.size == 10
    assert split_val.size == 10
    assert np.setdiff1d(split_trn, split_val).size == 10
    # Check balanced class-split
    for split_ids in [split_trn, split_val]:
        c_class0 = 0
        c_class1 = 0
        for idx in split_ids:
            y = labels[idx]
            if y == 0:
                c_class0 += 1
            else:
                c_class1 += 1
        assert c_class0 == c_class1