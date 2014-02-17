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
    echo "Setting up host ${instances[${i}]}."
    #  Kill any previous daemons on this host
    echo "Killing any previous daemons that might be running."
    ssh -i spike-keypair ubuntu@${instances[${i}]} " if [ \`ps axf | grep distribution | grep -v grep | wc -l\` -ne 0 ]; then a=\`ps axf | grep distribution | grep -v grep | awk '{print \"kill \" $1}'\`; fi"
    #  Start the daemon on this host
    ssh -f -i spike-keypair ubuntu@${instances[${i}]} "cd /home/ubuntu/spike/ && nohup python src/distributiond.py > /dev/null 2>&1 &" > /dev/null
    #  Add this host to the admin's remote hosts file
    ssh -i spike-keypair ubuntu@${admin_host} "cd /home/ubuntu/spike/test/ && echo \"${instances[${i}]}\" >> remote_hosts.txt"
    echo "Started daemon on host ${instances[${i}]} and added a reference to it from the admin (${admin_host})"
done

