## Attribution & license notices

- **`train/csi_train/model.py`** is an unofficial modified implementation based
  on `models/wisppn_resnet.py` from [geekfeiw/WiSPPN](https://github.com/geekfeiw/WiSPPN).
  Paper: Fei Wang, Stanislav Panev, Ziyi Dai, Jinsong Han, Dong Huang,
  *"Can WiFi Estimate Person Pose?"*, [arXiv:1904.00277](https://arxiv.org/abs/1904.00277) (2019).
  The upstream repository has no license file; copyright of the parts derived
  from it remains with the original authors. This repository uses them with
  attribution.
- The **teacher stage** automatically downloads RTMDet/RTMPose ONNX models from
  download.openmmlab.com ([OpenMMLab mmpose](https://github.com/open-mmlab/mmpose),
  Apache-2.0) on first run. The model files themselves are not included in this
  repository.
- The BODY-18 limb definition in `teacher/csi_teacher/qa.py` is the standard
  OpenPose skeleton topology (factual data).

License of this repository: [MIT](LICENSE). Note that the parts of
`train/csi_train/model.py` derived from the original remain under the original
authors' copyright, independent of this repository's license.

The source/additions boundary and unverified redistribution grant for the WiSPPN-derived portions are recorded in [NOTICE.md](NOTICE.md). Repository-level MIT does not replace those rights.
