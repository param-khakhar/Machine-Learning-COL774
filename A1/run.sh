data_dir=$2
strategy=$4
threads=$3
size=$1

if [[ ${strategy} == "1" ]]; then
    ./A2 $size $data_dir $threads $strategy
fi
if [[ ${strategy} == "2" ]]; then
    ./A2 $size $data_dir $threads $strategy
fi
if [[ ${strategy} == "3" ]]; then
    ./A2 $size $data_dir $threads $strategy
fi
if [[ ${strategy} == "4" ]]; then
    ./A2 $size $data_dir $threads $strategy
fi

