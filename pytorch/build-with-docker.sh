docker pull amazonlinux:1
docker run -v $(pwd):/outputs --name lambdapackgen-pytorch -d amazonlinux:1 tail -f /dev/null
docker exec -i -t lambdapackgen-pytorch /bin/bash /outputs/buildPack_py3.sh

