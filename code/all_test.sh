# 0830a: scale = [2,-1]
# 0831: scale = [1,0.01] getWorldObservation不强行改action=0
# 0831a: scale=[2, -1] no action changed
# 0901: scale=[2,-1] no compulsory action change
# 0902 : batch_size = 200

# python patrolling_task.py --T=5 --tau=5 --gridsize=5 --alg=CFPER --suffix=0905 # ok

# python charging_task.py --T=10 --tau=10 --gridsize=5 --alg=CFPER --suffix=0905
# python charging_task.py --T=10 --tau=20 --gridsize=5 --alg=CFPER --suffix=0905

# python patrolling_task.py --T=5 --tau=5 --gridsize=5 --alg=naiveER --suffix=0905

# python patrolling_task.py --T=15 --tau=15 --gridsize=5 --alg=CFPER --suffix=0905a

# python patrolling_task.py --T=20 --tau=20 --gridsize=10 --alg=naiveER --suffix=0905a
# python patrolling_task.py --T=20 --tau=20 --gridsize=10 --alg=CFPER --suffix=0905a




python patrolling_task.py --T=10 --tau=10 --gridsize=5 --alg=naiveER --suffix=0905


# python patrolling_task.py --T=20 --tau=20 --gridsize=10 --alg=naiveER --suffix=0905a
