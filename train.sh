for file in data/*.txt; do
    id=$(echo $file | cut -d'_' -f3 | cut -d'.' -f1)
    echo "training on $id"
    nkache --trace $file --ckpt-dir ./ckpt/$id/ --num-steps 100000 --num-episodes 50
done