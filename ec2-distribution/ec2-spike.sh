#! /bin/bash
set -e

NUM_INSTANCES=2

declare -a instance_ids
declare -a public_dns_names

wait_for_state ()
{
    local num_in_state=0
    local -A observed_instances=()

    for a in "${instance_ids[@]}"
    do
        observed_instances["${a}"]="not-in-state"
    done

    while true
    do
        sleep 1
        for ins_id in "${instance_ids[@]}"
        do
            state=`aws ec2 describe-instances --instance-ids "${ins_id}" | python -c "exec(\"import sys \nimport json \ndata = sys.stdin.readlines() \nobj=json.loads(''.join(data)) \nfor res in obj['Reservations']: \n for ins in res['Instances']: \n  if ins['InstanceId']: \n   print ins['State']['Name']\")"`
            echo "Instance ${ins_id} is in state ${state}"
            if [ "${state}" == "${1}" ] && [ "${observed_instances["${ins_id}"]}" == "not-in-state" ]
            then
                observed_instances["${ins_id}"]="in-state"
                num_in_state=$((num_in_state + 1))
                echo "Observed instance ${ins_id} in state ${state}"
            fi
        done

        if [ ${num_in_state} -eq ${NUM_INSTANCES} ]
        then
            break
        fi
        echo "${num_in_state} of ${NUM_INSTANCES} instances are in state ${1}"
    done
}

install_software_on_nodes () 
{
    for ins_id in "${instance_ids[@]}"
    do
        public_dns=`aws ec2 describe-instances --instance-ids "${ins_id}" | python -c "exec(\"import sys \nimport json \ndata = sys.stdin.readlines() \nobj=json.loads(''.join(data)) \nfor res in obj['Reservations']: \n for ins in res['Instances']: \n  if ins['InstanceId']: \n   print ins['PublicDnsName']\")"`
        public_dns_names+=("${public_dns}")
        echo "Obtained public DNS name of ${public_dns} from instance id ${ins_id}"
    done
}

create_cluster ()
{
    aws ec2 create-security-group --group-name spike-security-group --description "This is not a very secure security group."

    aws ec2 authorize-security-group-ingress --group-name spike-security-group --ip-permissions '{"FromPort":0,"ToPort":65535,"IpProtocol":"tcp","IpRanges":[{"CidrIp": "0.0.0.0/0"}]}'

    aws ec2 create-key-pair --key-name spike-keypair | python -c "import sys; import json; data = sys.stdin.readlines(); obj=json.loads(''.join(data)); print obj['KeyMaterial'];" > spike-keypair

    chmod go-rwx spike-keypair


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
    echo "${instance_ids[@]}" > instance_ids
}

delete_cluster ()
{
    #  Restore the instance ids from a file
    all_instance_ids=`cat instance_ids`
    instance_ids=( $all_instance_ids )
    for instance_id in "${instance_ids[@]}"
    do
        aws ec2 terminate-instances --instance-ids ${instance_id}
        echo "Issued termination command for instance ${instance_id}"
    done

    wait_for_state "terminated"

    aws ec2 delete-key-pair --key-name spike-keypair

    aws ec2 delete-security-group --group-name spike-security-group
}

if [ "${1}" == "create-cluster" ]
then
    create_cluster
elif [ "${1}" == "delete-cluster" ]
then
    delete_cluster
else
    echo "Usage: ./ec2-spike.sh [ create-cluster | delete-cluster ]"
fi 
