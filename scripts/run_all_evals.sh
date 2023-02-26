    for noise in aggressive
do
    for explain_method in gcexplainer protgnn 
    do
        for dataset in bashapes bacommunity
        do
            python src/pipeline.py --explain_method $explain_method --noise_method $noise --dataset $dataset --model_location models/${explain_method}_${dataset}.pt --output_location results/${explain_method}_${dataset}_${noise}.txt
        done
    done
done