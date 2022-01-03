train_data=$2
test_data=$3
out_file=$4
question=$1

if [[ ${question} == '1' ]]; then
    python3 q1a.py $train_data $test_data $out_file
fi
if [[ ${question} == '2' ]]; then
    python3 q1b.py $train_data $test_data $out_file
fi
if [[ ${question} == '3' ]]; then
    python3 q1c.py $train_data $test_data $out_file
fi
