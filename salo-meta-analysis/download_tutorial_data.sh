# A small script to download the data required for the NiMARE tutorial.
#
# This script only needs to be run if the data are not already available,
# such as if you are running the tutorial on your local machine.
#
# The data are located at https://osf.io/dnm2f/
#
# If you are running this tutorial locally, you will need to change the
# DATA_DIR variable in the notebook.

DIR="data/meta-analysis/"
if [ -d "$DIR" ]; then
    echo "$DIR exists."
else
    mkdir -p $DIR;
    pip install osfclient
    osf -p dnm2f clone  $DIR;
    echo "Created $DIR and downloaded the data";
fi
