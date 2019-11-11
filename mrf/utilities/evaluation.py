import typing

import numpy as np
import pymia.evaluation.evaluator as pymia_eval
import pymia.evaluation.metric as pymia_metric
import SimpleITK as sitk

import mrf.data.data as data


class Result:

    def __init__(self, subject: str, map_: str, metric: str, mask: str, value: float):
        self.subject = subject
        self.map_ = map_
        self.metric = metric
        self.mask = mask
        self.value = value

    def __eq__(self, other):
        return self.subject == other.subject and self.map_ == other.label and \
               self.metric == other.metric and self.mask == other.mask


class SummaryResult:

    def __init__(self, map_: str, metric: str, mask: str, mean: float, std: float):
        self.map_ = map_
        self.metric = metric
        self.mask = mask
        self.mean = mean
        self.std = std

    def __eq__(self, other):
        return self.map_ == other.label and self.metric == other.metric and self.mask == other.mask

    def __lt__(self, other):
        return self.map_[0] > other.map_[0] and self.metric[0] > other.metric[0]


def get_default_map_mask_combinations(maps: list):
    out = {}
    for map_ in maps:
        if map_ == data.FileTypes.T1H2Omap.name:
            # for water T1, a different mask is used to due the MRF T1-FF sequence's low confidence of water T1 at high FFs
            out[map_.replace('map', '')] = 'T1H2O'
        elif map_ == data.FileTypes.FFmap.name:
            out[map_.replace('map', '')] = 'FG'
        elif map_ == data.FileTypes.B1map.name:
            out[map_.replace('map', '')] = 'FG'
        else:
            raise ValueError('Map "{}" not supported'.format(map_.replace('map', '')))

    return out


class Evaluator:

    def __init__(self, writers: typing.List[pymia_eval.IEvaluatorWriter], metrics: typing.List[pymia_metric.IMetric],
                 maps: list, map_mask_combinations_fn=get_default_map_mask_combinations):
        self.writers = writers
        self.metrics = metrics
        self.maps = maps
        # order of maps in configuration will correspond to order of array extraction in data loading, and therefore to
        # order of prediction

        self.map_idx_combinations = {map_.replace('map', ''): idx for idx, map_ in enumerate(self.maps)}
        self.map_mask_combinations = map_mask_combinations_fn(self.maps)
        self.header = []
        self.results = []
        self.results_for_writers = []
        self._write_header()

    def evaluate(self, maps_prediction: np.ndarray, maps_reference: np.ndarray, masks: dict, subject_id: str):
        """Evaluates the desired map-mask combinations on a predicted subject.

        Args:
            maps_prediction: The predicted maps. Size is (Z, Y, X, N) where N is the number of maps.
            maps_reference: The reference maps. Size is (Z, Y, X, N) where N is the number of maps.
            masks: A dict with various masks of size (Z, Y, X). The dict keys identify the masks.
            subject_id: The subject identification.
        """

        # for masking we use np.extract, which works only on 1-D arrays --> reshape
        maps_prediction_flattened = np.reshape(maps_prediction, (-1, maps_prediction.shape[-1]))
        maps_reference_flattened = np.reshape(maps_reference, (-1, maps_reference.shape[-1]))
        masks_flattened = {mask_key: np.reshape(mask, (-1, mask.shape[-1])) for mask_key, mask in masks.items()}

        for map_id, mask_id in self.map_mask_combinations.items():
            map_results = [subject_id, mask_id, map_id]

            # mask
            map_idx = self.map_idx_combinations[map_id]
            map_prediction = np.extract(masks_flattened[mask_id] == 1, maps_prediction_flattened[..., map_idx])
            map_reference = np.extract(masks_flattened[mask_id] == 1, maps_reference_flattened[..., map_idx])
            map_prediction_img = sitk.GetImageFromArray(maps_prediction[..., map_idx])
            map_reference_img = sitk.GetImageFromArray(maps_reference[..., map_idx])

            for metric in self.metrics:
                if isinstance(metric, pymia_metric.INumpyArrayMetric):
                    metric.ground_truth = map_reference
                    metric.segmentation = map_prediction
                elif isinstance(metric, pymia_metric.ISimpleITKImageMetric):
                    metric.ground_truth = map_reference_img
                    metric.segmentation = map_prediction_img
                else:
                    raise NotImplementedError('Only INumpyArrayMetric and ISimpleITKImageMetric implemented')

                result = metric.calculate()
                map_results += [result, ]
                self.results.append(Result(subject_id, map_id, metric.metric, mask_id, result))

            self.results_for_writers.append(map_results)

    def write(self):
        for writer in self.writers:
            writer.write(self.results_for_writers)

    def _write_header(self):
        self.header = ['ID', 'MASK', 'MAP'] + [metric.metric for metric in self.metrics]
        for writer in self.writers:
            writer.write_header(self.header)

    def get_summaries(self) -> typing.List[SummaryResult]:
        summaries = []

        for map_id, mask_id in self.map_mask_combinations.items():
            for metric in self.metrics:
                results = [result.value for result in self.results if
                           result.map_ == map_id and result.mask == mask_id and result.metric == metric.metric]
                summaries.append(SummaryResult(map_id, metric.metric, mask_id,
                                               float(np.mean(results)), float(np.std(results))))
        return summaries


class SummaryResultWriter:

    def __init__(self, file_path: str = None, to_console: bool = True, precision: int = 3):
        self.file_path = file_path
        self.to_console = to_console
        self.precision = precision

    def write(self, data: typing.List[SummaryResult]):
        header = ['MAP', 'MASK', 'METRIC', 'RESULT']
        data = sorted(data)

        # we store the output data as list of list to nicely format the intends
        out_as_string = [header]
        for result in data:
            out_as_string.append([result.map_,
                                  result.mask,
                                  result.metric,
                                  '{0:.{2}f} ± {1:.{2}f}'.format(result.mean, result.std, self.precision)])

        # determine length of each column for output alignment
        lengths = np.array([list(map(len, row)) for row in out_as_string])
        lengths = lengths.max(0)
        lengths += (len(lengths) - 1) * [2] + [0, ]  # append two spaces except for last column

        # format for output alignment
        out = [['{0:<{1}}'.format(val, lengths[idx]) for idx, val in enumerate(line)] for line in out_as_string]

        to_print = '\n'.join(''.join(line) for line in out)
        if self.to_console:
            print(to_print)

        if self.file_path is not None:
            with open(self.file_path, 'w+') as file:
                file.write(to_print)
