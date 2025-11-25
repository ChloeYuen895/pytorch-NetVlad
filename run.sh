cd "C:\Users\yueny\OneDrive\Documents\Netvlad_3001\pytorch-NetVlad"
conda activate zoo

# Cluster (~10 min)
python main.py --mode cluster --arch vgg16 --pooling netvlad --num_clusters 64 --train_path zoo5/train --cache_path cache

# Train (~2 h 10 min)
python main.py --mode train ^
  --arch vgg16 --pooling netvlad --num_clusters 64 ^
  --resume pretrained/vd16_pitts30k_netvlad.pth ^
  --train_path zoo5/train --val_path zoo5/val ^
  --epochs 20 --batch_size 8 --lr 0.0001 --margin 0.1 ^
  --workers 6 --pin_memory ^
  --cache_path cache --save_path checkpoints