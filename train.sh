echo "The shell is running!"


python train.py --gpu_ids 0 --name "grcnn_rgb_depth" --model "ggcnn" --epoch 1000  --batch_size 8 