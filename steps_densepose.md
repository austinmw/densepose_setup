[GETTING STARTED](https://github.com/facebookresearch/DensePose/blob/master/GETTING_STARTED.md)

# Step 1: Get COCO data:

make a directory structure:  

coco/  
    annotations/  

#### train/val annotations
wget http://images.cocodataset.org/annotations/annotations_trainval2014.zip  

#### valminusminival2014, minival2014:
wget https://storage.googleapis.com/coco-dataset/external/PASCAL_VOC.zip  

#### images
cd path/to/coco/  
mkdir val2014 && gsutil -m rsync gs://images.cocodataset.org/val2014 val2014  
mkdir train2014 && gsutil -m rsync gs://images.cocodataset.org/train2014 train2014  

#### structure:   

/path/to/coco/  
    train2014  
    val2014  
    annotations/  
        instances_train2014.json  
        instances_val2014.json  
        instances_valminusminival2014.json  
        instances_minival2014.json  
        (not sure if keypoints/captions needed, prob not)  
    
    
    

# Step 2: Set up DensePose

## get densepose
DENSEPOSE=/home/austin/densepose  
git clone https://github.com/facebookresearch/densepose $DENSEPOSE  

## get densepose annotations
cd $DENSEPOSE/DensePoseData  
bash get_densepose_uv.sh  
bash get_DensePose_COCO.sh  
bash get_eval_data.sh  

## set up docker
cd $DENSEPOSE/docker  
docker build -t densepose:c2-cuda9-cudnn7 .  

nvidia-docker run --rm -it densepose:c2-cuda9-cudnn7 python2 detectron/tests/test_batch_permutation_op.py  

## modify docker

#### enter container
nvidia-docker run -v $DENSEPOSE/DensePoseData:/denseposedata -v /home/austin/coco:/coco -it densepose:c2-cuda9-cudnn7 bash  
mv /densepose/DensePoseData /densepose/DensePoseDataLocal  
ln -s /denseposedata DensePoseData  
ln -s /coco /densepose/detectron/datasets/data/coco  
ln -s /densepose/DensePoseData/DensePose_COCO/densepose_coco_2014_minival.json /densepose/detectron/datasets/data/coco/annotations/  
ln -s /densepose/DensePoseData/DensePose_COCO/densepose_coco_2014_train.json /densepose/detectron/datasets/data/coco/annotations/  
ln -s /densepose/DensePoseData/DensePose_COCO/densepose_coco_2014_valminusminival.json /densepose/detectron/datasets/data/coco/annotations/  

#### Exit container and commit changes
docker commit $(docker ps --last 1 -q) densepose:c2-cuda9-cudnn7-wdata  




# Part 3: Run inference/testing/training commands
`nvidia-docker run --rm -v $DENSEPOSE/DensePoseData:/denseposedata -v /home/austin/coco:/coco -it densepose:c2-cuda9-cudnn7-wdata <inference_or_training_command>`


#### testing
nvidia-docker run --rm -v $DENSEPOSE/DensePoseData:/denseposedata -v /home/austin/coco:/coco -it densepose:c2-cuda9-cudnn7-wdata python2 tools/test_net.py --cfg configs/DensePose_ResNet101_FPN_s1x-e2e.yaml TEST.WEIGHTS https://s3.amazonaws.com/densepose/DensePose_ResNet101_FPN_s1x-e2e.pkl NUM_GPUS 1

#### inference

##### Single image inference on demo data
nvidia-docker run --rm -v $DENSEPOSE/DensePoseData:/denseposedata -v /home/austin/coco:/coco -it densepose:c2-cuda9-cudnn7-wdata \
python2 tools/infer_simple.py \
    --cfg configs/DensePose_ResNet101_FPN_s1x-e2e.yaml \
    --output-dir DensePoseData/infer_out/ \
    --image-ext jpg \
    --wts https://s3.amazonaws.com/densepose/DensePose_ResNet101_FPN_s1x-e2e.pkl \
    DensePoseData/demo_data/demo_im.jpg

##### Multiple image inference on own data
DENSEPOSE=/home/austin/densepose && nvidia-docker run -v $DENSEPOSE/DensePoseData:/denseposedata -v /home/austin/coco:/coco  -v /home/austin/data/densepose_mydata:/densepose_mydata -it -p 8888:8888 --name densepose densepose:c2-cuda9-cudnn7-wdata \
    python2 tools/infer_simple.py \
    --cfg configs/DensePose_ResNet101_FPN_s1x-e2e.yaml \
    --output-dir /densepose_mydata/infer_out/ \
    --image-ext jpg \
    --wts https://s3.amazonaws.com/densepose/DensePose_ResNet101_FPN_s1x-e2e.pkl \
    /densepose_mydata/test_imgs/


Part 4: Visualization of outputs (don't run with --rm if want to visualize in jupyter)

jupyter notebook --ip 0.0.0.0 --no-browser --allow-root  
notebooks/DensePose-RCNN-Visualize-Results.ipynb  
loc: /densepose_mydata/infer_out







# Script

if i make a script, try using popd/pushd  
can just set everything as detached mode: -d  
