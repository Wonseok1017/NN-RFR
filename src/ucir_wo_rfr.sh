device_id=0
SEED=0
bz=128
lr=0.1
mom=0.9
wd=5e-4
data=cifar100_icarl
network=resnet18_cifar
nepochs=160
n_exemplar=20

appr=rfr_ucir
lamb=5.0

nc_first=50
ntask=6 # For S=10
# ntask=11 # For S=5
# ntask=26 # For S=2


# UCIR w/o RFR
rfr_coef=0.00

CUDA_VISIBLE_DEVICES=$device_id python3 main_incremental.py --exp-name seed_${SEED}_nc_first_${nc_first}_ntask_${ntask}_${rfr_coef} \
    --datasets $data --num-tasks $ntask --nc-first-task $nc_first --network $network --seed $SEED --nepochs $nepochs \
    --batch-size $bz --lr $lr --momentum $mom --weight-decay $wd --decay-mile-stone 80 120 --clipping -1 \
    --save-models --approach $appr --lamb $lamb --num-exemplars-per-class $n_exemplar \
    --exemplar-selection herding --rfr-coef $rfr_coef
