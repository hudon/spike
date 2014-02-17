#! /bin/bash
set -e


dns_names=`cat public_dns_names`

instances=( ${dns_names} ) 

#  Set up the admin
admin_host=${instances[0]}

#  Truncate the old remote hosts file
ssh -i spike-keypair ubuntu@${admin_host} "cd /home/ubuntu/spike/test/ && :> remote_hosts.txt"

# fill up the remote hosts file with references to the new nodes
for (( i=1; i<${#instances[@]}; i++ ))
do
    #  Start the daemon on this host
    ssh -f -i spike-keypair ubuntu@${instances[${i}]} "cd /home/ubuntu/spike/ && nohup python src/distributiond.py > /dev/null 2>&1 &" > /dev/null
    #  Add this host to the admin's remote hosts file
    ssh -i spike-keypair ubuntu@${admin_host} "cd /home/ubuntu/spike/test/ && echo \"${instances[${i}]}\" >> remote_hosts.txt"
    echo "Started daemon on host ${instances[${i}]} and added a reference to it from the admin (${admin_host})"
done

#  Run the unit tests
ssh -i spike-keypair ubuntu@${admin_host} "cd /home/ubuntu/spike/ && make test"
