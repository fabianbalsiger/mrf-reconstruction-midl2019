import os

import numpy as np
import pymia.data.assembler as pymia_asmbl
import pymia.data.conversion as pymia_conv
import pymia.data.definition as pymia_def
import pymia.deeplearning.data_handler as hdlr
import pymia.deeplearning.model as mdl
import pymia.deeplearning.testing as test
import pymia.evaluation.evaluator as pymia_eval
import SimpleITK as sitk
import tensorflow as tf

import mrf.data.data as data
import mrf.utilities.assembler as asmbl
import mrf.utilities.evaluation as eval
import mrf.utilities.metric as metric
import mrf.utilities.normalization as norm
import mrf.utilities.plt_qualitative as plt


def process_predictions(self: test.Tester, subject_assembler: pymia_asmbl.Assembler, result_dir, map_list):
    arr_idx_to_map_name_dict = {idx: map_list[idx].replace('map', '')
                                for idx in range(len(map_list))}

    os.makedirs(result_dir, exist_ok=True)
    csv_file = os.path.join(result_dir, 'results.csv')
    evaluator = eval.Evaluator([pymia_eval.CSVEvaluatorWriter(csv_file)], metric.get_metrics(), map_list)

    # loop over all subjects
    for subject_idx in list(subject_assembler.predictions.keys()):
        subject_data = self.data_handler.dataset.direct_extract(self.data_handler.extractor_test, subject_idx)
        subject_name = subject_data['subject']

        # rescale and mask reference maps (clipping will have no influence)
        maps = norm.process(subject_data[pymia_def.KEY_LABELS], subject_data[data.ID_MASK_FG],
                            subject_data['norm'], map_list)

        # rescale, clip, and mask prediction
        prediction = subject_assembler.get_assembled_subject(subject_idx)
        prediction = np.reshape(prediction, subject_data[pymia_def.KEY_SHAPE] + (prediction.shape[-1],))
        prediction = norm.process(prediction, subject_data[data.ID_MASK_FG], subject_data['norm'], map_list)

        # evaluate on foreground
        masks = {'FG': subject_data[data.ID_MASK_FG], 'T1H2O': subject_data[data.ID_MASK_T1H2O]}
        evaluator.evaluate(prediction, maps, masks, subject_name)

        # Save predictions as SimpleITK images and save other images
        subject_results = os.path.join(result_dir, subject_name)
        os.makedirs(subject_results, exist_ok=True)
        plotter = plt.QualitativePlotter(subject_results, 2, 'png')

        for map_idx, map_name in enumerate(map_list):
            map_name_short = map_name.replace('map', '')
            # save predicted maps
            prediction_image = pymia_conv.NumpySimpleITKImageBridge.convert(prediction[..., map_idx],
                                                                            subject_data[pymia_def.KEY_PROPERTIES])
            sitk.WriteImage(prediction_image,
                            os.path.join(subject_results, '{}_{}.mha'.format(subject_name, map_name_short)),
                            True)

            plotter.plot(subject_name, map_name, prediction[..., map_idx], maps[..., map_idx],
                         subject_data[data.ID_MASK_T1H2O] if map_name == data.FileTypes.T1H2Omap.name
                         else subject_data[data.ID_MASK_FG])

    evaluator.write()


class MRFTensorFlowTester(test.TensorFlowTester):

    def __init__(self, data_handler: hdlr.DataHandler, model: mdl.TensorFlowModel, model_dir: str, result_dir: str,
                 maps, session: tf.Session):
        super().__init__(data_handler, model, model_dir, session)
        self.result_dir = result_dir
        self.maps = maps

    def init_subject_assembler(self) -> pymia_asmbl.Assembler:
        return asmbl.init_subject_assembler()

    def process_predictions(self, subject_assembler: pymia_asmbl.Assembler):
        process_predictions(self, subject_assembler, self.result_dir, self.maps)

    def batch_to_feed_dict(self, batch: dict):
        feed_dict = {self.model.x_placeholder: np.stack(batch[pymia_def.KEY_IMAGES], axis=0),
                     self.model.is_training_placeholder: False}
        return feed_dict
