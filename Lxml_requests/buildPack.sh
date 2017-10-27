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
        libjpeg-devel \
        zlib-devel \
        zip
    ln -s /usr/libexec/gcc/x86_64-amazon-linux/4.8.5/cc1plus /usr/local/bin/
}

pip_rasterio () {
    cd /home/
    virtualenv env
    source env/bin/activate

    pip install --upgrade pip wheel
    pip install --use-wheel lxml
    pip install --use-wheel requests
}


gather_pack () {
    # packing
    cd /home/
    mkdir lambdapack
    cd lambdapack

    cp -R $VIRTUAL_ENV/lib/python2.7/site-packages/* .
    cp -R $VIRTUAL_ENV/lib64/python2.7/site-packages/* .
    echo "original size $(du -sh /home/lambdapack | cut -f1)"

    # cleaning
    find -name "*.so" | xargs strip
    find -name "*.so.*" | xargs strip
    rm -r pip
    rm -r pip-9.0.1.dist-info
    rm -r wheel
    rm -r wheel-0.30.0.dist-info
    rm easy_install.py
    find . -name \*.pyc -delete
    echo "stripped size $(du -sh /home/lambdapack | cut -f1)"

    # compressing
    zip -r9 /outputs/pack.zip * > /dev/null
    echo "compressed size $(du -sh /outputs/pack.zip | cut -f1)"
}

main () {
    dev_install

    pip_rasterio

    gather_pack
}

main