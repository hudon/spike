HOSTS=(
    "ctngpu1"
    "ctngpu2"
    "ctngpu3"
);


sortkeys() {
    echo "$1" | tr " " "\n" | sort | tr "\n" " "
}

for i in {1..999999};
do
    declare -A row
    declare -A previouscpudata
    declare -A totalmem

    for host in "${HOSTS[@]}";
    do
        a=`ssh -i /home/spike/.ssh/serverkey spike@$host.uwaterloo.ca -p 3656 "ssh root@localhost -p 3123 \"cat /proc/meminfo && cat /proc/stat\""`
        cached=`echo "$a" | grep ^Cached: | grep -o "[0-9]\+"`
        memfree=`echo "$a" | grep ^MemFree: | grep -o "[0-9]\+"`
        memtotal=`echo "$a" | grep ^MemTotal: | grep -o "[0-9]\+"`
        totalmem[${host}]=$memtotal
        row[${host}-ava]=$(($cached + $memfree))

        cpualluser=`echo "$a" | grep "cpu " | awk '{print $2}'`
        cpuallnice=`echo "$a" | grep "cpu " | awk '{print $3}'`
        cpuallsystem=`echo "$a" | grep "cpu " | awk '{print $4}'`
        cpuallidle=`echo "$a" | grep "cpu " | awk '{print $5}'`
        cpualliowait=`echo "$a" | grep "cpu " | awk '{print $6}'`
        cpuallirq=`echo "$a" | grep "cpu " | awk '{print $7}'`
        cpuallsoftirq=`echo "$a" | grep "cpu " | awk '{print $8}'`
        totalcpuallused=$(($cpualluser + $cpuallnice + $cpuallsystem + $cpualliowait + $cpuallirq + $cpuallsoftirq))
        totalcpuallidle=$(($cpuallidle))
        #  used / (used + idle)
        numerator=$(($totalcpuallused - previouscpudata[${host}-cpu]))
        denominator=$((  ( ($totalcpuallused - previouscpudata[${host}-cpu]) + ($totalcpuallidle - previouscpudata[${host}-idle]) )  ))
        row[${host}-util]=`echo "scale=2; $numerator/$denominator" | bc`
        previouscpudata[${host}-cpu]=$totalcpuallused 
        previouscpudata[${host}-idle]=$totalcpuallidle
    done


    if [ $i -eq 1 ] ; then
        eval keys=\${!totalmem[@]}
        sorted_keys=$(sortkeys "$keys")

        for key in $sorted_keys
        do
             echo "$key has ${totalmem[$key]} total memory"
        done
    fi


    echo -n `date`
    echo -ne "\t"

    eval keys=\${!row[@]}
    sorted_keys=$(sortkeys "$keys")

    for key in $sorted_keys;
    do
        if [ $i -eq 1 ] ; then
            printf "%20s" "$key"
        else
            printf "%20s" "${row[$key]}"
        fi
    done

    echo ""
    sleep 10
done
