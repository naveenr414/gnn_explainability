
    for noise in aggressive conservative
do
    for frac in 0.0
  do
      for explain_method in cdm
      do
          for dataset in bashapes
          do

            if [ "$explain_method" = "gcexplainer" ]
            then

              for k_means in fixed varied
              do
              python3 src/pipeline.py --explain_method $explain_method --noise_method $noise --noise_amount $frac --dataset $dataset --model_location models/${explain_method}_${dataset}.pt --output_location ${explain_method}_${dataset}_${noise}_${frac}_${k_means}
              done

            else
              python3 src/pipeline.py --explain_method $explain_method --noise_method $noise --noise_amount $frac --dataset $dataset --model_location models/${explain_method}_${dataset}.pt --output_location ${explain_method}_${dataset}_${noise}_${frac}
              fi

          done
      done
  done
done