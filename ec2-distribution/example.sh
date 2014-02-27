#! /bin/bash
set -e

#  Example that creates a cluster, sets up the network topology, performs a simulation, then deletes the cluster.

echo "Creating an EC2 cluster with 3 nodes..."
./ec2-spike.sh create-cluster 3


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
    ssh -i spike-keypair ubuntu@${instances[${i}]} "/home/ubuntu/spike/ec2-distribution/kill-daemon.sh"
    #  Start the daemon on this host
    ssh -f -i spike-keypair ubuntu@${instances[${i}]} "cd /home/ubuntu/spike/ && nohup python src/distributiond.py > /dev/null 2>&1 &" > /dev/null
    #  Add this host to the admin's remote hosts file
    ssh -i spike-keypair ubuntu@${admin_host} "cd /home/ubuntu/spike/test/ && echo \"${instances[${i}]}\" >> remote_hosts.txt"
    echo "Started daemon on host ${instances[${i}]} and added a reference to it from the admin (${admin_host})"
done

echo "Starting a simulation on admin host ${admin_host}."
echo "First 20 lines of output from simulation:"
ssh -i spike-keypair ubuntu@${admin_host} "/usr/bin/python2 /home/ubuntu/spike/test/nengo_tests/test_array.py    /home/ubuntu/spike/test/../src --hosts=/home/ubuntu/spike/test/remote_hosts.txt" | sed -n 1,20p
echo "Finished a simulation on admin host ${admin_host}."
echo "Deleting cluster..."

./ec2-spike.sh delete-cluster

echo "Cluster has been deleted successfully.  Example complete."
