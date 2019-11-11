import numpy as np
import pymia.data.definition as pymia_def
import pymia.data.transformation as pymia_tfm


class MRFLabelNormalization(pymia_tfm.Transform):

    def __init__(self, min_: dict, max_: dict) -> None:
        super().__init__()
        self.min_ = min_
        self.max_ = max_

    def __call__(self, sample: dict) -> dict:
        maps = pymia_tfm.check_and_return(sample[pymia_def.KEY_LABELS], np.ndarray)

        for idx, entry in enumerate(self.min_.keys()):
            maps[..., idx] = (maps[..., idx] - self.min_[entry]) / (self.max_[entry] - self.min_[entry])  # normalize to 0..1

        sample[pymia_def.KEY_LABELS] = maps
        return sample


class MRFMaskedLabelNormalization(pymia_tfm.Transform):

    def __init__(self, min_: dict, max_: dict, mask: str) -> None:
        super().__init__()
        self.min_ = min_
        self.max_ = max_
        self.mask = mask

    def __call__(self, sample: dict) -> dict:
        # normalize first
        norm = MRFLabelNormalization(self.min_, self.max_)
        sample = norm(sample)

        maps = pymia_tfm.check_and_return(sample[pymia_def.KEY_LABELS], np.ndarray)
        mask = pymia_tfm.check_and_return(sample[self.mask], np.ndarray)

        mask = np.concatenate([mask] * maps.shape[-1], -1)
        maps[mask == 0] = 0

        sample[pymia_def.KEY_LABELS] = maps
        return sample


class MRFTemporalOcclusion(pymia_tfm.Transform):
    """Temporal occlusion"""

    def __init__(self, temporal_idx: int=0, entries=(pymia_def.KEY_IMAGES, )):
        super().__init__()
        self.temporal_idx = temporal_idx
        self.entries = entries

    def __call__(self, sample: dict) -> dict:
        for entry in self.entries:
            if entry not in sample:
                raise ValueError(pymia_tfm.ENTRY_NOT_EXTRACTED_ERR_MSG.format(entry))

            np_entry = pymia_tfm.check_and_return(sample[entry], np.ndarray)
            np_entry[..., self.temporal_idx, :] = 0
            sample[entry] = np_entry

        return sample
