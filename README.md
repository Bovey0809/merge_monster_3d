# Merge Monster of Yolo & PointNet++ & CenterNet3D

Extract 2D image features from Yolo.
Extract 3D features from PointNet++.
Combine them as input for CenterNet3D's Head.
Boom!

## TODOS

- [x] PointNet feature extractor.
- [x] 2D images feature map extractor.
- [x] Single Training (mainly for debugging).
- [x] Distribute Training
- [ ] Inference (For 3D Objects detection)
- [ ] Evaluation (Metric Map)

## Basic Script

### Train 

- Single GPU(Debugging)
    ```bash
    python tools/train.py configs/mergenet/merge_net.py
    ```
- Multi gpus(8 gpus)
    ```bash
    ./tools/dist_train.sh configs/mergenet/merge_net.py 8
    ```

### Test

- Single GPU(demo visualization)
    ```bash
    python tools/test.py configs/mergenet/merge_net.py weights/lastest.pth
    ```
- Multi GPUS()
### 