# On the Spatial and Temporal Influence for the Reconstruction of Magnetic Resonance Fingerprinting
This repository contains code for the [MIDL 2019](https://2019.midl.io/) paper "On the Spatial and Temporal Influence for the Reconstruction of Magnetic Resonance Fingerprinting", which can be found at https://openreview.net/forum?id=HyeuSq9ke4.


## Installation

The installation has been tested with Ubuntu 18.04, Python 3.6, TensorFlow 1.10, and CUDA 9.0. The ``setup.py`` file lists all other dependencies.

First, create a virtual environment named `mrf` with Python 3.6:

        $ virtualenv --python=python3.6 mrf
        $ source ./mrf/bin/activate

Second, copy the code:

        $ git clone https://github.com/fabianbalsiger/mrf-reconstruction-midl2019
        $ cd mrf-reconstruction-midl2019

Third, install the required libraries:

        $ pip install -r requirements.txt

This should install all required dependencies. Please refer to the official TensorFlow documentation on how to use TensorFlow with [GPU support](https://www.tensorflow.org/install/gpu).

## Usage

We shortly describe the training and testing procedure.
The data used in the paper is not publicly available. But, we provide a script to generate dummy data such that you are able to run the code.

### Dummy Data Generation and Configuration

We handle the data using [pymia](https://pymia.readthedocs.io/en/latest). Therefore, we need to create a hierarchical data format (HDF) file to have easy and fast access to our data during the training and testing.
Create the dummy data by

        $ python ./snippets/create_dataset.py

This will create the file ``./data/data.h5``, or simply our dataset. Use any open source HDF viewer to inspect the file (e.g., [HDFView](https://www.hdfgroup.org/downloads/hdfview/)).
Please refer to the [pymia documentation](https://pymia.readthedocs.io/en/latest/examples.dataset.html) on how to create your own dataset. 

We now create so called indices files, which allow fast data access. Execute

        $ python ./snippets/create_indices_files.py

This will create a JSON file per subject in the directory ``./bin/indices``. Each index basically references a 32 x 32 patch in the images, which we can access through array slicing.

Further, we need to create a training/validation/testing split file by

        $ python ./snippets/create_split.py

This will create the file ``./bin/split.json``. Note that you need to create a similar file when using your own dataset.

### Training
To train the model, simply execute ``./bin/main.py``. The data and training parameters are provided by the ``./bin/config.json``, which you can adapt to your needs.
Note that you might want to specify the CUDA device by

        $ CUDA_VISIBLE_DEVICES=0 python ./bin/main.py

The script will automatically use the training subjects defined in ``./bin/split.json`` and evaluate the model's performance after every 5th epoch on the validation subjects.
The validation will be saved under the path ``result_dir`` specified in the configuration file ``./bin/config.json``.
The trained model will be saved under the path ``model_dir`` specified in the configuration file ``./bin/config.json``.
Further, the script logs the training and validation progress for visualization using [TensorBoard](https://www.tensorflow.org/guide/summaries_and_tensorboard).
Start the TensorBoard to observe the training:

        $ tensorboard --logdir=<path to the model_dir>

### Testing
The training script (``./bin/main.py``) directly provides you with the reconstructions of the validation subjects.
So, you need to execute the testing script only when you want apply it to your testing set.

        $ CUDA_VISIBLE_DEVICES=0 python ./bin/test.py --model_dir=<patch to your model directory> --result_dir="<path to your result directory>"

#### Occlusion Experiments
To perform the occlusion experiments, use

        $ CUDA_VISIBLE_DEVICES=0 python ./bin/test.py --model_dir=<patch to your model directory> --result_dir="<path to your result directory>" --do_occlusion=True

Note that this might take a while to run.

## Support
We leave an explanation of the code as exercise ;-). But if you found a bug or have a specific question, please open an issue or a pull request.

Generally, adaptions to your MRF sequence should be straight forward. Once you were able to generate the HDF file, the indices files, and the split file, the code should run by itself without massive modifications. 

## Citation

If you use this work, please cite

```
Balsiger, F., Scheidegger, O., Carlier, P. G., Marty, B., & Reyes, M. (2019). On the Spatial and Temporal Influence for the Reconstruction of Magnetic Resonance Fingerprinting. In M. J. Cardoso, A. Feragen, B. Glocker, E. Konukoglu, I. Oguz, G. Unal, & T. Vercauteren (Eds.), Proceedings of The 2nd International Conference on Medical Imaging with Deep Learning (pp. 27–38). London: PMLR.
```

```
@InProceedings{pmlr-v102-balsiger19a,
  title = {On the Spatial and Temporal Influence for the Reconstruction of Magnetic Resonance Fingerprinting},
  author = {Balsiger, Fabian and Scheidegger, Olivier and Carlier, Pierre G. and Marty, Benjamin and Reyes, Mauricio},
  booktitle = {Proceedings of The 2nd International Conference on Medical Imaging with Deep Learning},
  pages = {27--38},
  year = {2019},
  editor = {Cardoso, M. Jorge and Feragen, Aasa and Glocker, Ben and Konukoglu, Ender and Oguz, Ipek and Unal, Gozde and Vercauteren, Tom},
  volume = {102},
  series = {Proceedings of Machine Learning Research},
  address = {London, United Kingdom},
  month = {08--10 Jul},
  publisher = {PMLR},
  pdf = {http://proceedings.mlr.press/v102/balsiger19a/balsiger19a.pdf},
  url = {http://proceedings.mlr.press/v102/balsiger19a.html},
}
```

## License

The code is published under the [MIT License](https://github.com/fabianbalsiger/mrf-reconstruction-midl2019/blob/master/LICENSE).