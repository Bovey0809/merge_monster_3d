# Merge Monster of Yolo & PointNet++ & CenterNet3D

Extract 2D image features from Yolo.
Extract 3D features from PointNet++.
Combine them as input for CenterNet3D's Head.
Boom!

## Build Env

NOTE: Please download and process the data according to 
[mmdetction3d dataset prepare](https://mmdetection3d.readthedocs.io/en/latest/data_preparation.html).

```bash
git clone https://github.com/Bovey0809/merge_monster_3d.git
cd merge_monster_3d
docker build --network=host -f docker/Dockerfile -t mergenet .
docker run --gpus all --ipc=host -it -v $HOME/data:/mmdetection3d/data -v $HOME/work_dirs/:/mmdetection3d/work_dirs mergenet
```

## TODOS

- [x] PointNet feature extractor.
- [x] 2D images feature map extractor.
- [x] Single Training (mainly for debugging).
- [x] Distribute Training
- [x] Inference (For 3D Objects detection)
- [x] Evaluation (Metric Map)
- [x] NanoDet train, test, eval

## Basic Script

### Train 

- Single GPU(Debugging)
    ```bash
    python tools/train.py configs/mergenet/merge_net.py
    python tools/train.py configs/nanodet/nanodet.py
    ```
- Multi gpus(8 gpus)
    ```bash
    ./tools/dist_train.sh configs/mergenet/merge_net.py 8
    ./tools/dist_train.sh configs/nanodet/nanodet.py 8
    ```

### Test

- Single GPU(demo visualization)
    ```bash
    python tools/test.py configs/mergenet/merge_net.py weights/lastest.pth
    ```
- Multi GPUS()
    ```bash
    ./tools/dist_test.sh configs/nanodet/nanodet.py weights.pth 8
    ```
    or
    ```bash
    CUDA_VISIBLE_DEVICES=1,2,3,4 ./tools/dist_test.sh configs/nanodet/nanodet.py weights.pth 4
    ```
    
### 
