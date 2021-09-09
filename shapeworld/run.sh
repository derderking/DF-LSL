HYPO_LAMBDA=20

python df-lsl/train.py --cuda \
    --predict_concept_hyp \
    --hypo_lambda $HYPO_LAMBDA \
    --batch_size 100 \
    --seed $RANDOM \
    exp/df-lsl