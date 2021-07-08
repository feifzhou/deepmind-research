## tensorflow 1.x
# conda activate powerai


DIR=learning_to_simulate
python -m $DIR.train \
    --data_path=$DIR/data/GNN-hydrodynamics/WaterRamps/ \
    --model_path=$DIR/experiments/WaterRamps/models/ \
    --output_path=$DIR/experiments/WaterRamps/rollouts \
    --batch_size=14 --lr_decay=100000


# cloth
python -m meshgraphnets.run_model --model=cloth --mode=train \
    --checkpoint_dir=meshgraphnets/experiment --dataset_dir=$HOME/data/airfoil/flag_minimal  --num_training_steps=100000
python -m meshgraphnets.run_model --model=cloth --mode=eval \
    --checkpoint_dir=meshgraphnets/experiment --dataset_dir=$HOME/data/airfoil/flag_minimal  \
    --rollout_path=meshgraphnets/experiment/rollout.pkl --num_rollouts=1
python -m meshgraphnets.plot_cloth --rollout_path=meshgraphnets/experiment/rollout.pkl

# cylinder_flow
DIR=meshgraphnets/experiment/cylinder_flow
DAT=$HOME/data/airfoil/cylinder_flow
python -m meshgraphnets.run_model --model=cfd --mode=train --checkpoint_dir=$DIR --dataset_dir=$DAT --num_training_steps=100000 --batch=16
python -m meshgraphnets.run_model --model=cfd --mode=eval  --checkpoint_dir=$DIR --dataset_dir=$DAT --rollout_path=$DIR/rollout.pkl --num_rollouts=1 
python -m meshgraphnets.plot_cfd --rollout_path=$DIR/rollout.pkl
# output format:
#   faces[ntime, nface, 3] int
#   mesh_pos[ntime, nnode, dim] float
#   pred_velocity,pred_velocity[ntime, nnode, nfeature] float

# wave 1 mode
DIR=meshgraphnets/experiment/wave1mode
DAT=$HOME/data/waves-1mode
for i in valid train test; do
    python meshgraphnets/npy2tfrecord.py $DAT/$i.npy -o $DAT/$i;
done
python -m meshgraphnets.run_model --model=NPS --mode=train --checkpoint_dir=$DIR --dataset_dir=$DAT --nfeat_in=1 --nfeat_out=1 --num_training_steps=100000 --batch=8  
python -m meshgraphnets.run_model --model=NPS --mode=eval  --checkpoint_dir=$DIR --dataset_dir=$DAT --nfeat_in=1 --nfeat_out=1 --rollout_path=$DIR/rollout.pkl --num_rollouts=1
python -m meshgraphnets.plot_cfd --rollout_path=$DIR/rollout.pkl



# grain growth
DIR=meshgraphnets/experiment/grain
DAT=$HOME/data/grain
for i in valid train test; do
    python meshgraphnets/npy2tfrecord.py $DAT/$i.npy -o $DAT/$i --periodic;
done
python -m meshgraphnets.run_model --model=NPS --mode=train --checkpoint_dir=$DIR --dataset_dir=$DAT --periodic=1 --nfeat_in=1 --nfeat_out=1 --num_training_steps=500000 --batch=8
python -m meshgraphnets.run_model --model=NPS --mode=eval  --checkpoint_dir=$DIR --dataset_dir=$DAT --periodic=1 --nfeat_in=1 --nfeat_out=1 --rollout_split=test --rollout_path=$DIR/rollout.pkl --num_rollouts=1
python -m meshgraphnets.plot_cfd --rollout_path=$DIR/rollout.pkl --skip=5 --mirrory
