declare -A PORTS
PORTS["ctngpu1"]="3656"
PORTS["ctngpu2"]="3656"
PORTS["ctngpu3"]="3656"
PORTS["ctn08"]="22"
PORTS["129.97.45.78"]="22"

declare -A POSTFIX
POSTFIX["ctngpu1"]=".uwaterloo.ca"
POSTFIX["ctngpu2"]=".uwaterloo.ca"
POSTRIX["ctngpu3"]=".uwaterloo.ca"
POSTFIX["ctn08"]=".uwaterloo.ca"
POSTFIX["129.97.45.78"]=""

HOSTS=(
    "ctngpu1"
    "ctngpu2"
    "ctngpu3"
    "ctn08"
    "129.97.45.78"
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
        a=`ssh -i /home/spike/.ssh/serverkey spike@${host}${POSTFIX[${host}]} -p ${PORTS[${host}]} "cat /proc/meminfo && cat /proc/stat"`
        cached=`echo "$a" | grep ^Cached: | grep -o "[0-9]\+"`
        memfree=`echo "$a" | grep ^MemFree: | grep -o "[0-9]\+"`
        memtotal=`echo "$a" | grep ^MemTotal: | grep -o "[0-9]\+"`
        totalmem["${host}"]=$memtotal
        row["${host}""-ava"]=$(($cached + $memfree))

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
        numerator=$(($totalcpuallused - previouscpudata["${host}""-cpu"]))
        denominator=$((  ( ($totalcpuallused - previouscpudata["${host}""-cpu"]) + ($totalcpuallidle - previouscpudata["${host}""-idle"]) )  ))
        row["${host}""-util"]=`echo "scale=2; $numerator/$denominator" | bc`
        previouscpudata["${host}""-cpu"]=$totalcpuallused 
        previouscpudata["${host}""-idle"]=$totalcpuallidle
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
            printf "%18s" "$key"
        else
            printf "%18s" "${row[$key]}"
        fi
    done

    echo ""
    sleep 10
done
