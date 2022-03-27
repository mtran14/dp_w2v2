The code is mainly based on the [S3PRL Speech Toolkit](https://github.com/s3prl/s3prl). Please refer to the S3PRL toolkit for installation instructions. A set of trained models used in the project can be found [here](https://drive.google.com/drive/folders/1giu-gg6Vp-iSzik0-9KpMg0h8DlregqR?usp=sharing).

1. Train the speaker identification  (SID) model. For dataset preparation, please follow this [link](https://github.com/s3prl/s3prl/blob/master/s3prl/downstream/docs/superb.md#sid-speaker-identification).
```
cd s3prl
python3 run_downstream.py -m train -u wav2vec2 -d voxceleb1 -n sid_w2v2
```

2. Generate the Privacy-risk Saliency map dataset.
```
cd s3prl
python3 run_downstream_saliency_map.py -m train -e results/downstream/sid_w2v2/dev-best.ckpt --save_path path_to_store_saliecy_maps
```

3. Train the Privacy-risk Saliency Estimator model.
```
cd s3prl
python3 train_pse.py path_to_store_saliecy_maps
```

4. Apply pertubations on the downstream tasks. Please follow the instructions [here](https://github.com/s3prl/s3prl/blob/master/s3prl/downstream/docs/superb.md) to prepare the datasets.
```
cd s3prl
python3 run_downstream_perturb.py -m evaluate -e path_to_trained_downstream_model --pse path_to_trained_pse --eps EPS --threshold THRESHOLD
```
THRESHOLD: {0 (100% perturbed), 1 (80% perturbed), 2 (60% perturbed), 3 (40% perturbed), 4 (20% perturbed)}

EPS: amount of perturbation (e.g. 0.5, 1.0, etc.)

If you use the provided downstream models, consider using `-o config.downstream_expert.datarc.[file_path/root/libri_root/]` to set the dataset paths correctly.
