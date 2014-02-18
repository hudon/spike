#! /bin/bash
set -e

dns_names=`cat public_dns_names`

instances=( ${dns_names} ) 

#  Set up the admin
admin_host=${instances[0]}

#  Truncate the old remote hosts file
ssh -i spike-keypair ubuntu@${admin_host} "cd /home/ubuntu/spike/test/ && :> remote_hosts.txt"
ssh -i spike-keypair ubuntu@${admin_host} "sudo rm -f /tmp/*"

# fill up the remote hosts file with references to the new nodes
for (( i=1; i<${#instances[@]}; i++ ))
do
    echo "Setting up host ${instances[${i}]}."
    #  Kill any previous daemons on this host
    ssh -i spike-keypair ubuntu@${instances[${i}]} "cd /home/ubuntu/spike && git pull origin ec2-distribution"
    #ssh -i spike-keypair ubuntu@${instances[${i}]} "/home/ubuntu/spike/ec2-distribution/kill-daemon.sh"
    #  Start the daemon on this host
    #ssh -f -i spike-keypair ubuntu@${instances[${i}]} "cd /home/ubuntu/spike/ && nohup python src/distributiond.py > /dev/null 2>&1 &" > /dev/null
    #  Add this host to the admin's remote hosts file
    ssh -i spike-keypair ubuntu@${admin_host} "cd /home/ubuntu/spike/test/ && echo \"${instances[${i}]}\" >> remote_hosts.txt"
    echo "Started daemon on host ${instances[${i}]} and added a reference to it from the admin (${admin_host})"
done

