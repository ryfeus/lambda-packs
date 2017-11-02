dev_install () {
    yum -y update
    yum -y upgrade
    yum install -y \
        wget \
        gcc \
        gcc-c++ \
        python27-devel \
        python27-virtualenv \
        python27-pip \
        findutils \
        zlib-devel \
        zip
}

pip_rasterio () {
    cd /home/
    virtualenv env
    source env/bin/activate

    pip install --upgrade pip wheel
    pip install google -U
    pip install --use-wheel numpy -U
    pip install --use-wheel tensorflow -U

    deactivate
}


gather_pack () {
    # packing
    cd /home/
    source env/bin/activate

    rm -r lambdapack
    mkdir lambdapack
    cd lambdapack

    cp -R $VIRTUAL_ENV/lib/python2.7/site-packages/* .
    cp -R $VIRTUAL_ENV/lib64/python2.7/site-packages/* .
    cp /outputs/index.py /home/lambdapack/index.py
    echo "original size $(du -sh /home/lambdapack | cut -f1)"

    # cleaning libs
    rm -r external
    find . -type d -name "tests" -exec rm -rf {} +

    # cleaning
    find -name "*.so" | xargs strip
    find -name "*.so.*" | xargs strip
    # find . -name tests -type d -print0|xargs -0 rm -r --
    # find . -name test -type d -print0|xargs -0 rm -r --    
    rm -r pip
    rm -r pip-9.0.1.dist-info
    rm -r wheel
    rm -r wheel-0.30.0.dist-info
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