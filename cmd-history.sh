Local
```
python train_test_split.py C:\Users\Bo.Sun\repos\objection-detection\workspace\annotations\coordinates.csv -f 0.8 -o C:\Users\Bo.Sun\repos\objection-detection\workspace\annotations
python csv_to_TFRecord.py --csv_input=C:\Users\Bo.Sun\repos\objection-detection\workspace\annotations\coordinates_train.csv --img_path=C:\Users\Bo.Sun\repos\objection-detection\workspace\asset\train_imgs --output_path=C:\Users\Bo.Sun\repos\objection-detection\workspace\annotations\coord_train.record
python csv_to_TFRecord.py --csv_input=C:\Users\Bo.Sun\repos\objection-detection\workspace\annotations\coordinates_eval.csv --img_path=C:\Users\Bo.Sun\repos\objection-detection\workspace\asset\eval_imgs --output_path=C:\Users\Bo.Sun\repos\objection-detection\workspace\annotations\coord_eval.record
```

Cloud 
```
source ~/setup.sh

pip install pycocotools

python object_detection/model_main.py --pipeline_config_path=/home/ec2-user/obj-detection/pre-trained-models/ssd_resnet50_v1_fpn_shared_box_predictor_640x640_coco14_sync_2018_07_03/pipeline.config --model_dir=/home/ec2-user/obj-detection/fine-tuned-model --num_train_steps=25000 --sample_1_of_n_eval_examples=1 --alsologtostderr

tensorboard --logdir=~/obj-detection/asset/


```
python export_inference_graph.py --input_type image_tensor --pipeline_config_path /home/ec2-user/obj-detection/pre-trained-models/ssd_resnet50_v1_fpn_shared_box_predictor_640x640_coco14_sync_2018_07_03/pipeline.config --trained_checkpoint_prefix /home/ec2-user/obj-detection/fine-tuned-model/model.ckpt-24433 --output_directory /home/ec2-user/obj-detection/output/model




# trained with 100 images 
python export_inference_graph.py --input_type image_tensor --pipeline_config_path /home/ec2-user/obj-detection/pre-trained-models/ssd_resnet50_v1_fpn_shared_box_predictor_640x640_coco14_sync_2018_07_03/pipeline.config --trained_checkpoint_prefix /home/ec2-user/obj-detection/asset/model.ckpt-23534 --output_directory /home/ec2-user/obj-detection/test/model

python inference_v2.py /home/ec2-user/obj-detection/eval_images/ /home/ec2-user/obj-detection/test/ --model=/home/ec2-user/obj-detection/test/model/frozen_inference_graph.pb


# train with 1000 images

python object_detection/model_main.py --pipeline_config_path=/home/ec2-user/obj-detection/pre-trained-models/ssd_resnet50_v1_fpn_shared_box_predictor_640x640_coco14_sync_2018_07_03/pipeline.config --model_dir=/home/ec2-user/obj-detection/fine-tuned-model --num_train_steps=15000 --sample_1_of_n_eval_examples=1 --alsologtostderr

tensorboard --logdir=~/obj-detection/fine-tuned-model/


## SSD 1000 images, continue 20000:

python object_detection/model_main.py --pipeline_config_path=/home/ec2-user/obj-detection/pre-trained-models/ssd_resnet50_v1_fpn_shared_box_predictor_640x640_coco14_sync_2018_07_03/pipeline.config --model_dir=/home/ec2-user/obj-detection/fine-tuned-model --num_train_steps=30000 --sample_1_of_n_eval_examples=1 --logtostderr

python export_inference_graph.py --input_type image_tensor --pipeline_config_path /home/ec2-user/obj-detection/pre-trained-models/ssd_resnet50_v1_fpn_shared_box_predictor_640x640_coco14_sync_2018_07_03/pipeline.config --trained_checkpoint_prefix /home/ec2-user/obj-detection/fine-tuned-model/model.ckpt-????? --output_directory /home/ec2-user/obj-detection/SSDoutput/model


python inference_v2.py /home/ec2-user/obj-detection/eval_images/ /home/ec2-user/obj-detection/SSDoutput --model=/home/ec2-user/obj-detection/SSDoutput/model/frozen_inference_graph.pb --no_save_crop



# FasterCNN

python label_csv_to_tfRecord.py --csv_input=/home/ec2-user/obj-detection/annotations/coordinates.csv --img_path=/home/ec2-user/obj-detection/train_images --output_path=/home/ec2-user/obj-detection/annotations/coordinates.record

python object_detection/model_main.py --pipeline_config_path=/home/ec2-user/obj-detection/pre-trained-models/faster_rcnn_inception_resnet_v2_atrous_oid_v4_2018_12_12/pipeline.config --model_dir=/home/ec2-user/obj-detection/FasterCNN-log --num_train_steps=50000 --sample_1_of_n_eval_examples=1 --logtostderr







# yuqi rcnn export
python export_inference_graph.py --input_type encoded_image_string_tensor --pipeline_config_path /home/ec2-user/v2-obj-detection/fastrcnn-yuqi/pipeline.config --trained_checkpoint_prefix /home/ec2-user/v2-obj-detection/fastrcnn-yuqi/model.ckpt-150000 --output_directory /home/ec2-user/v2-obj-detection/fastrcnn-yuqi-export









# better-image, model-02
python label_csv_to_tfRecord.py --csv_input=/home/ec2-user/v2-obj-detection/train-job-02/annotations/coordinates.csv --img_path=/home/ec2-user/v2-obj-detection/train-job-02/train-images --output_path=/home/ec2-user/v2-obj-detection/train-job-02/annotations/coordinates.record


python object_detection/model_main.py --pipeline_config_path=/home/ec2-user/v2-obj-detection/train-job-03/annotations/pipeline.config --model_dir=/home/ec2-user/v2-obj-detection/train-job-03/model-train-with-normalize --num_train_steps=30000 --sample_1_of_n_eval_examples=1 --logtostderr


python export_inference_graph.py --input_type image_tensor --pipeline_config_path=/home/ec2-user/v2-obj-detection/train-job-02/annotations/pipeline.config --trained_checkpoint_prefix /home/ec2-user/v2-obj-detection/train-job-02/model-train-from-scratch/model.ckpt-27500 --output_directory /home/ec2-user/v2-obj-detection/TFServing/batch-inference




python export_inference_graph.py --input_type image_tensor --pipeline_config_path=/home/ec2-user/v2-obj-detection/fastrcnn-yuqi/pipeline.config --trained_checkpoint_prefix /home/ec2-user/v2-obj-detection/fastrcnn-yuqi/model.ckpt-150000 --output_directory /home/ec2-user/v2-obj-detection/fastrcnn-yuqi-export-imarr


python inference.py /home/ec2-user/obj-detection/eval_images/ /home/ec2-user/obj-detection/Faster-RCNN-output --model=/home/ec2-user/v2-obj-detection/fastrcnn-yuqi-export/frozen_inference_graph.pb --image_tensor_type=string --no_save_crop

python inference_string_tensor.py /home/ec2-user/obj-detection/eval_images/ /home/ec2-user/v2-obj-detection/train-job-03/inference-output-string --model=/home/ec2-user/v2-obj-detection/train-job-03/export-string-tensor/frozen_inference_graph.pb --input_tensor_type=string --no_save_crop





python export_inference_graph.py \
    --input_type encoded_image_string_tensor \
    --pipeline_config_path /home/ec2-user/v2-obj-detection/fastrcnn-yuqi/pipeline.config \
    --trained_checkpoint_prefix /home/ec2-user/v2-obj-detection/fastrcnn-yuqi/model.ckpt-150000 \
    --output_directory /home/ec2-user/v2-obj-detection/frcnn-export \
    --config_override " \
            model{ \
              faster_rcnn { \
                second_stage_post_processing { \
                  batch_non_max_suppression { \
                    score_threshold: 0.5 \
                    max_total_detections: 100 \
                  } \
                } \
              } \
            }"
            
            
python export_inference_graph.py \
    --input_type encoded_image_string_tensor \
    --pipeline_config_path /home/ec2-user/v2-obj-detection/train-job-03/annotations/pipeline.config \
    --trained_checkpoint_prefix /home/ec2-user/v2-obj-detection/train-job-03/model-train-with-normalize/model.ckpt-30000 \
    --output_directory /home/ec2-user/v2-obj-detection/train-job-03/export-string-tensorl