import os
import torch.multiprocessing as mp
import torch.distributed as dist


def add_ddp_args(parser):
    parser.add_argument('-n', '--nodes', default=1, type=int, metavar='N')
    parser.add_argument('-g', '--gpus', default=1, type=int,
                        help='number of gpus per node')
    parser.add_argument('--nr', default=0, type=int,
                        help='ranking within the nodes')
    parser.add_argument('--ip', default='localhost', type=str,
                        help='IP Address of host node')
    parser.add_argument('--port', default="8888", type=str,
                        help='port')
    return parser


def ddp_setup(main, args):
    args.world_size = args.gpus * args.nodes
    os.environ['MASTER_ADDR'] = args.ip
    os.environ['MASTER_PORT'] = args.port
    os.environ['NCCL_DEBUG'] = "WARN"
    mp.spawn(main, nprocs=args.gpus, args=(args,))
    return args


def init_process(gpu, args):
    args.rank = args.nr * args.gpus + gpu
    dist.init_process_group(
        backend='nccl',
        init_method='env://',
        world_size=args.world_size,
        rank=args.rank
    )
    return args
