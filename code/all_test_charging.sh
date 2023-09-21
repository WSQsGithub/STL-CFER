cd task

SUFFIX=$1

echo "SUFFIX=$SUFFIX"

# fundamental
# python charging_task.py --T=10 --tau=5 --gridsize=5 --alg=naive --suffix="$SUFFIX"

# python charging_task.py --T=10 --tau=5 --gridsize=5 --alg=naiveER --suffix="$SUFFIX"

# python charging_task.py --T=10 --tau=5 --gridsize=5 --alg=CFER --suffix="$SUFFIX"

# python charging_task.py --T=10 --tau=5 --gridsize=5 --alg=CFPER --suffix="$SUFFIX"

# changing tau = 7
# python charging_task.py --T=10 --tau=7 --gridsize=5 --alg=naive --suffix="$SUFFIX"

# python charging_task.py --T=10 --tau=7 --gridsize=5 --alg=naiveER --suffix="$SUFFIX"

# python charging_task.py --T=10 --tau=7 --gridsize=5 --alg=CFER --suffix="$SUFFIX"

# python charging_task.py --T=10 --tau=7 --gridsize=5 --alg=CFPER --suffix="$SUFFIX"

# changing tau = 9
# python charging_task.py --T=10 --tau=9 --gridsize=5 --alg=naive --suffix="$SUFFIX"

# python charging_task.py --T=10 --tau=9 --gridsize=5 --alg=naiveER --suffix="$SUFFIX"

# python charging_task.py --T=10 --tau=10 --gridsize=5 --alg=CFER --suffix="$SUFFIX"

# python charging_task.py --T=10 --tau=20 --gridsize=5 --alg=CFER --suffix="$SUFFIX"

# python charging_task.py --T=10 --tau=9 --gridsize=5 --alg=CFPER --suffix="$SUFFIX"

# changing resolution to 10*10
# python charging_task.py --T=10 --tau=5 --gridsize=10 --alg=naive --suffix="$SUFFIX"

# python charging_task.py --T=10 --tau=5 --gridsize=10 --alg=naiveER --suffix="$SUFFIX"

# python charging_task.py --T=10 --tau=5 --gridsize=10 --alg=CFER --suffix="$SUFFIX"

# python charging_task.py --T=10 --tau=5 --gridsize=10 --alg=CFPER --suffix="$SUFFIX"

# python charging_task.py --T=10 --tau=20 --gridsize=10 --alg=naive --suffix="$SUFFIX"
# python charging_task.py --T=10 --tau=10 --gridsize=10 --alg=naive --suffix="$SUFFIX"
# python charging_task.py --T=10 --tau=5 --gridsize=10 --alg=naive --suffix="$SUFFIX"

# 4000 round
python charging_task.py --T=10 --tau=20 --gridsize=5 --alg=naiveER --suffix="$SUFFIX"
python charging_task.py --T=10 --tau=10 --gridsize=5 --alg=naiveER --suffix="$SUFFIX"
python charging_task.py --T=10 --tau=5 --gridsize=5 --alg=naiveER --suffix="$SUFFIX"
python charging_task.py --T=10 --tau=20 --gridsize=10 --alg=naiveER --suffix="$SUFFIX"
python charging_task.py --T=10 --tau=10 --gridsize=10 --alg=naiveER --suffix="$SUFFIX"
python charging_task.py --T=10 --tau=5 --gridsize=10 --alg=naiveER --suffix="$SUFFIX"