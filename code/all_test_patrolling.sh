# 0904: exp all neg reward, no scaling, min_eps = 0.01

cd task

SUFFIX=$1

echo "SUFFIX=$SUFFIX"

# patrolling test
# fundamentals
# python patrolling_task.py --T=10 --tau=10 --gridsize=5 --alg=naive --suffix="$SUFFIX" 

# python patrolling_task.py --T=10 --tau=10 --gridsize=5 --alg=naiveER --suffix="$SUFFIX"

# python patrolling_task.py --T=10 --tau=10 --gridsize=5 --alg=CFER --suffix="$SUFFIX"

# python patrolling_task.py --T=5 --tau=5 --gridsize=5 --alg=naive --suffix="$SUFFIX"
# python patrolling_task.py --T=5 --tau=5 --gridsize=5 --alg=CFER --suffix="$SUFFIX"



# python patrolling_task.py --T=10 --tau=10 --gridsize=10 --alg=naive --suffix="$SUFFIX"
# python patrolling_task.py --T=20 --tau=20 --gridsize=10 --alg=naive --suffix="$SUFFIX"
# python patrolling_task.py --T=20 --tau=20 --gridsize=5 --alg=naive --suffix="$SUFFIX"
# python patrolling_task.py --T=15 --tau=15 --gridsize=10 --alg=naive --suffix="$SUFFIX"
# python patrolling_task.py --T=15 --tau=15 --gridsize=5 --alg=naive --suffix="$SUFFIX"


# python patrolling_task.py --T=10 --tau=10 --gridsize=10 --alg=naive --suffix="$SUFFIX"

# python patrolling_task.py --T=20 --tau=20 --gridsize=10 --alg=CFER --suffix="$SUFFIX"
# python patrolling_task.py --T=20 --tau=20 --gridsize=10 --alg=naive --suffix="$SUFFIX"

# python patrolling_task.py --T=20 --tau=20 --gridsize=5 --alg=CFER --suffix="$SUFFIX"
# python patrolling_task.py --T=20 --tau=20 --gridsize=5 --alg=naive --suffix="$SUFFIX"

# python patrolling_task.py --T=10 --tau=10 --gridsize=5 --alg=CFPER --suffix="$SUFFIX"

# tau = 7
# python patrolling_task.py --T=10 --tau=7 --gridsize=5 --alg=naive --suffix="$SUFFIX" 

# python patrolling_task.py --T=10 --tau=7 --gridsize=5 --alg=naiveER --suffix="$SUFFIX"

# python patrolling_task.py --T=10 --tau=7 --gridsize=5 --alg=CFER --suffix="$SUFFIX"

# python patrolling_task.py --T=10 --tau=7 --gridsize=5 --alg=CFPER --suffix="$SUFFIX"

# # tau = 9
# python patrolling_task.py --T=10 --tau=9 --gridsize=5 --alg=naive --suffix="$SUFFIX" 

# python patrolling_task.py --T=10 --tau=9 --gridsize=5 --alg=naiveER --suffix="$SUFFIX"

# python patrolling_task.py --T=10 --tau=9 --gridsize=5 --alg=CFER --suffix="$SUFFIX"

# python patrolling_task.py --T=10 --tau=9 --gridsize=5 --alg=CFPER --suffix="$SUFFIX"

# # grid = 10
# python patrolling_task.py --T=10 --tau=9 --gridsize=10 --alg=naive --suffix="$SUFFIX" 

# python patrolling_task.py --T=10 --tau=9 --gridsize=10 --alg=naiveER --suffix="$SUFFIX"

# python patrolling_task.py --T=10 --tau=9 --gridsize=10 --alg=CFER --suffix="$SUFFIX"

# python patrolling_task.py --T=10 --tau=9 --gridsize=10 --alg=CFPER --suffix="$SUFFIX"

python patrolling_task.py --T=15 --tau=15 --gridsize=10 --alg=naiveER --suffix="$SUFFIX"
python patrolling_task.py --T=10 --tau=10 --gridsize=10 --alg=naiveER --suffix="$SUFFIX"
python patrolling_task.py --T=5 --tau=5 --gridsize=10 --alg=naiveER --suffix="$SUFFIX"
python patrolling_task.py --T=15 --tau=15 --gridsize=5 --alg=naiveER --suffix="$SUFFIX"
python patrolling_task.py --T=10 --tau=10 --gridsize=5 --alg=naiveER --suffix="$SUFFIX"
python patrolling_task.py --T=5 --tau=5 --gridsize=5 --alg=naiveER --suffix="$SUFFIX"