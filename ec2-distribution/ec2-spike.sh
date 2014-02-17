#! /bin/bash
set -e


declare -a instance_ids
declare -A public_dns_names

sortkeys() {
    echo "$1" | tr " " "\n" | sort | tr "\n" " "
}

wait_for_state ()
{
    local num_in_state=0
    local -A observed_instances=()

    for a in "${instance_ids[@]}"
    do
        observed_instances["${a}"]="not-in-state"
    done

    eval keys=\${instance_ids[@]}
    local -a sorted_instance_ids=$(sortkeys "$keys")

    echo ""

    for header in $sorted_instance_ids
    do
        printf "%18s" "${header}"
    done
    echo ""

    while true
    do
        local -a state_progress=()
        sleep 1
        for ins_id in $sorted_instance_ids
        do
            if [ "${1}" == "installed" ]
            then
                sleep 8
                num_scripts_running=`ssh ubuntu@${public_dns_names["${ins_id}"]} -i spike-keypair "ps -ef" | grep ec2-node-install | wc -l`
                if [ ${num_scripts_running} -eq 0 ]
                then
                    state="installed"
                else
                    state="not-installed"
                fi
            else
                state=`aws ec2 describe-instances --instance-ids "${ins_id}" | python -c "exec(\"import sys \nimport json \ndata = sys.stdin.readlines() \nobj=json.loads(''.join(data)) \nfor res in obj['Reservations']: \n for ins in res['Instances']: \n  if ins['InstanceId']: \n   print ins['State']['Name']\")"`
            fi

            state_progress+=("${state}")

            if [ "${state}" == "${1}" ] && [ "${observed_instances["${ins_id}"]}" == "not-in-state" ]
            then
                observed_instances["${ins_id}"]="in-state"
                num_in_state=$((num_in_state + 1))
            fi
        done

        for s in "${state_progress[@]}"
        do
            printf "%18s" "${s}"
        done
        echo -e "\t${num_in_state} of ${NUM_INSTANCES} instances are in state ${1}"

        if [ ${num_in_state} -eq ${NUM_INSTANCES} ]
        then
            break
        fi
    done
}

install_software_on_nodes () 
{
    for ins_id in "${instance_ids[@]}"
    do
        public_dns=`aws ec2 describe-instances --instance-ids "${ins_id}" | python -c "exec(\"import sys \nimport json \ndata = sys.stdin.readlines() \nobj=json.loads(''.join(data)) \nfor res in obj['Reservations']: \n for ins in res['Instances']: \n  if ins['InstanceId']: \n   print ins['PublicDnsName']\")"`
        public_dns_names["${ins_id}"]="${public_dns}"
        echo "Obtained public DNS name of ${public_dns} from instance id ${ins_id}"
        #  The first time we attempt to ssh in, the instance might not be visible yet, so we have to retry
        echo "Installing git and cloning repo on host ${public_dns} with instance id ${ins_id}..."
        while true
        do
            ssh -oStrictHostKeyChecking=no ubuntu@${public_dns} -i spike-keypair "sudo apt-get install git -y && git clone https://github.com/Hudon/spike.git && cd spike && git branch ec2-distribution && git checkout ec2-distribution && git pull origin ec2-distribution" > /dev/null 2>&1 && if [ $? -eq 0 ]; then break; fi
            sleep 5
            echo "ssh to node ${public_dns} failed. retrying (this is usually expected)"
        done
        echo "ssh to node ${public_dns} was successful.  Initiating installation of spike software stack..."

        ssh ubuntu@${public_dns} -f -i spike-keypair "cd spike/ec2-distribution && nohup ./ec2-node-install.sh > /dev/null 2>&1 &" > /dev/null
    done
    #  Wait for all the software to install before returning to the user
    wait_for_state "installed"
}

create_cluster ()
{
    echo "Creating security group 'spike-security-group'..."
    aws ec2 create-security-group --group-name spike-security-group --description "This is not a very secure security group." > /dev/null
    echo "Authorizing ingress for security group 'spike-security-group'..."
    aws ec2 authorize-security-group-ingress --group-name spike-security-group --ip-permissions '{"FromPort":0,"ToPort":65535,"IpProtocol":"tcp","IpRanges":[{"CidrIp": "0.0.0.0/0"}]}' > /dev/null
    echo "Creating keypair 'spike-keypair'..."
    aws ec2 create-key-pair --key-name spike-keypair | python -c "import sys; import json; data = sys.stdin.readlines(); obj=json.loads(''.join(data)); print obj['KeyMaterial'];" > spike-keypair

    chmod go-rwx spike-keypair


    echo "Launching ${NUM_INSTANCES} instances..."
    for (( i=1; i<=${NUM_INSTANCES}; i++ ))
    do
        instance_info=`aws ec2 run-instances --image-id ami-d9a98cb0 --count 1 --instance-type t1.micro --key-name spike-keypair --security-groups spike-security-group`
        instance_id=`echo -n "$instance_info" | python -c "import sys; import json; data = sys.stdin.readlines(); obj=json.loads(''.join(data)); print obj['Instances'][0]['InstanceId'];"`
        instance_ids+=("${instance_id}")
        echo "Launched instance ${instance_id}"
    done

    wait_for_state "running"

    install_software_on_nodes
    #  Save the instance ids for when we want to delete them
    echo "Instance ids output to file 'instance_ids'"
    echo "${instance_ids[@]}" > instance_ids
    echo "Public DNS Names output to file 'public_dns_names'"
    echo "${public_dns_names[@]}" > public_dns_names
    echo "Cluster with ${NUM_INSTANCES} nodes has been created successfully."
}

delete_cluster ()
{
    #  Restore the instance ids from a file
    all_instance_ids=`cat instance_ids`
    instance_ids=( $all_instance_ids )
    for instance_id in "${instance_ids[@]}"
    do
        aws ec2 terminate-instances --instance-ids ${instance_id} > /dev/null
        echo "Issued termination command for instance ${instance_id}"
    done

    wait_for_state "terminated"
    echo "Deleting spike-keypair..."
    aws ec2 delete-key-pair --key-name spike-keypair > /dev/null
    echo "Deleting spike-security-group..."
    aws ec2 delete-security-group --group-name spike-security-group > /dev/null
    echo "Cluster has been deleted successfully."
}

if [ "${1}" == "create-cluster" ]
then
    if [ $# -ne 2 ]
    then
        echo "There should be 2 arguments.  ./ec2-spike.sh create-cluter [NUM_NODES]"
    else
        NUM_INSTANCES=${2}
        create_cluster
    fi
elif [ "${1}" == "delete-cluster" ]
then
    #  The number of instances will be the number of spaces between instance ids, plus one
    NUM_INSTANCES=`cat instance_ids | grep -o " " | wc -l`
    NUM_INSTANCES=$((NUM_INSTANCES + 1))
    delete_cluster
else
    echo "Usage: ./ec2-spike.sh [ create-cluster | delete-cluster ]"
fi 
