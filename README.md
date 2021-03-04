# Tensorflow-API-Mask-R-CNN（C++）
使用labelme标注数据时，将格式转换成COCO格式，然后保存为TFRecord格式，最用进行训练

1、labelme标签转coco
python labelme2coco.py train train.json
python labelme2coco.py test test.json

2、转换TFRecord格式
python create_coco_tf_record.py --logtostderr --train_image_dir=train --test_image_dir=test --train_annotations_file=train.json --test_annotations_file=test.json --output_dir=./

3、训练
python model_main.py --logtostderr --model_dir=model --pipeline_config_path=config/mask_rcnn_inception_v2_coco.config

4、导出
python export_inference_graph.py --input_type image_tensor --pipeline_config_path config/mask_rcnn_inception_v2_coco.config --trained_checkpoint_prefix model/model.ckpt-94332 --output_directory model/inference_graph

5、C++(OpenCV)调用需要生成pbtxt
python tf_text_graph_mask_rcnn.py --input model\inference_graph\frozen_inference_graph_build.pb --output model\inference_graph\graph.pbtxt --config config\mask_rcnn_inception_v2_coco.config