for x in snappy leveldb gflags glog szip hdf5 lmdb homebrew/science/opencv;
do
    brew uninstall $x;
    brew install --fresh -vd $x;
done
brew uninstall --force protobuf; brew install --with-python --fresh -vd protobuf
brew uninstall boost boost-python; brew install --fresh -vd boost boost-python

