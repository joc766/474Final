TestModel:
	echo "#!/bin/bash" > TestModel
	echo "python test_cribbage.py \"\$$@\"" >> TestModel
	chmod u+x TestModel

CreateModel:
	echo "#!/bin/bash" > CreateModel
	echo "python train.py \"\$$@\"" >> CreateModel
	chmod u+x CreateModel
