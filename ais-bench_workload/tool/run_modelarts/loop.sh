loop=0
count=0
#while [ $count –lt 100 ]
while [ $loop -lt 1 ]
do
    echo "job run $count begin"
    date
    python3 -u run_modelarts.py --timeout 200
    ret=$?
    date
    echo "job run $count end ret:$?"
    
    sleep 100
    count=$[$count+1]
done

