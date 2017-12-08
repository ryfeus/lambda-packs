dev_install () {
    yum -y update
    yum -y upgrade
    yum install -y \
    wget \
    gcc \
    gcc-c++ \
    python36-devel \
    python36-virtualenv \
    python36-pip \
    findutils \
    zlib-devel \
    zip
}

pip_rasterio () {
    cd /home/
    rm -rf env
    python3 -m virtualenv env --python=python3
    source env/bin/activate
    pip install -U pip wheel
    pip install --use-wheel numpy -U
    pip install --use-wheel tensorflow -U
    deactivate
}


gather_pack () {
    # packing
    cd /home/
    source env/bin/activate

    rm -rf lambdapack
    mkdir lambdapack
    cd lambdapack

    cp -R /home/env/lib/python3.6/site-packages/* .
    cp -R /home/env/lib64/python3.6/site-packages/* .
    cp /outputs/index_py3.py /home/lambdapack/index_py3.py
    echo "original size $(du -sh /home/lambdapack | cut -f1)"

    # cleaning libs
    rm -rf external
    find . -type d -name "tests" -exec rm -rf {} +

    # cleaning
    find -name "*.so" | xargs strip
    find -name "*.so.*" | xargs strip
    # find . -name tests -type d -print0|xargs -0 rm -r --
    # find . -name test -type d -print0|xargs -0 rm -r --    
    rm -r pip
    rm -r pip-*
    rm -r wheel
    rm -r wheel-*
    rm easy_install.py
    find . -name \*.pyc -delete
    # find . -name \*.txt -delete
    echo "stripped size $(du -sh /home/lambdapack | cut -f1)"

    # compressing
    zip -FS -r9 /outputs/pack.zip * > /dev/null
    echo "compressed size $(du -sh /outputs/pack.zip | cut -f1)"
}

main () {
    dev_install
    pip_rasterio
    gather_pack
}

main
