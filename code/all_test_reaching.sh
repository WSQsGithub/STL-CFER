cd task

SUFFIX=$1

echo "SUFFIX=$SUFFIX"

# reaching test
# fundamentals
# python reaching_task.py --T=10 --tau=5 --gridsize=5 --alg=naive --suffix="$SUFFIX" 

# python reaching_task.py --T=10 --tau=5 --gridsize=5 --alg=naiveER --suffix="$SUFFIX"

python reaching_task.py --T=10 --tau=5 --gridsize=5 --alg=naiveER --suffix="$SUFFIX"
python reaching_task.py --T=10 --tau=10 --gridsize=5 --alg=naiveER --suffix="$SUFFIX"
python reaching_task.py --T=10 --tau=20 --gridsize=5 --alg=naiveER --suffix="$SUFFIX"

python reaching_task.py --T=10 --tau=5 --gridsize=10 --alg=naiveER --suffix="$SUFFIX"
python reaching_task.py --T=10 --tau=10 --gridsize=10 --alg=naiveER --suffix="$SUFFIX"
python reaching_task.py --T=10 --tau=20 --gridsize=10 --alg=naiveER --suffix="$SUFFIX"


# python reaching_task.py --T=10 --tau=5 --gridsize=5 --alg=CFPER --suffix="$SUFFIX"


# # tau = 7
# python reaching_task.py --T=10 --tau=7 --gridsize=5 --alg=naive --suffix="$SUFFIX" 

# python reaching_task.py --T=10 --tau=7 --gridsize=5 --alg=naiveER --suffix="$SUFFIX"

# python reaching_task.py --T=10 --tau=7 --gridsize=5 --alg=CFER --suffix="$SUFFIX"

# python reaching_task.py --T=10 --tau=7 --gridsize=5 --alg=CFPER --suffix="$SUFFIX"

# # tau = 9
# python reaching_task.py --T=10 --tau=9 --gridsize=5 --alg=naive --suffix="$SUFFIX" 

# python reaching_task.py --T=10 --tau=9 --gridsize=5 --alg=naiveER --suffix="$SUFFIX"

# python reaching_task.py --T=10 --tau=9 --gridsize=5 --alg=CFER --suffix="$SUFFIX"

# python reaching_task.py --T=10 --tau=9 --gridsize=5 --alg=CFPER --suffix="$SUFFIX"

# # grid = 10
# python reaching_task.py --T=10 --tau=5 --gridsize=10 --alg=naive --suffix="$SUFFIX" 

# python reaching_task.py --T=10 --tau=5 --gridsize=10 --alg=naiveER --suffix="$SUFFIX"

# python reaching_task.py --T=10 --tau=5 --gridsize=10 --alg=CFER --suffix="$SUFFIX"

# python reaching_task.py --T=10 --tau=5 --gridsize=10 --alg=CFPER --suffix="$SUFFIX"


# python reaching_task.py --T=20 --tau=13 --gridsize=10 --alg=CFER --suffix="$SUFFIX"
# python reaching_task.py --T=20 --tau=13 --gridsize=10 --alg=CFER --suffix="$SUFFIX"


# 0905: shrinking batch size 