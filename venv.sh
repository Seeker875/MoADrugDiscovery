
set -x

#vars
srcDir=$(cd $(dirname $0); echo $PWD)
venvDir=$srcDir/venv


#create env
python3 -m venv $venvDir || exit$?
source $venvDir/bin/activate || exit $?

###Install pip
if ! [-f $srcDir/get-pip.py ] ; then
	wget https://bootstrap.pypa.io/get-pip.py  || exit $?
fi

python3 $srcDir/get-pip.py || exit $?
pip3 install --upgrade pip || exit $?

pip3 install tensorflow numpy matplotlib ipython jupyter pandas seaborn xgboost scikit-learn 
