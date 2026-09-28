import argparse
import json
from pathlib import Path
import subprocess
import sys


def main():
    p=argparse.ArgumentParser()
    for key in ('config','diagnostics','monitor','calibration','target-train','test','output'):
        p.add_argument('--'+key,required=True)
    p.add_argument('--hard-test');p.add_argument('--harp-root');p.add_argument('--delta-root');p.add_argument('--twohop-data')
    p.add_argument('--initial');p.add_argument('--rounds',type=int,default=100)
    p.add_argument('--preparation-costs',nargs='*',default=[])
    p.add_argument('--seeds',nargs='+',type=int,default=list(range(5)))
    p.add_argument('--methods',nargs='+',default=['reprsi','direct','prompted','matched-search','target','uncertainty','shuffled','human'])
    p.add_argument('--resume',action='store_true')
    args=p.parse_args();root=Path(args.output)
    if not args.methods or args.methods[0]!='reprsi':p.error('reprsi must run first to establish the per-seed compute budget')
    if len(set(args.seeds))!=len(args.seeds) or len(set(args.methods))!=len(args.methods):p.error('Duplicate seeds or methods')
    def invoke(parts):subprocess.run([sys.executable,'-m','reprsi',*map(str,parts)],check=True)
    dependencies=[]
    if args.harp_root:dependencies+=['--harp-root',args.harp_root]
    if args.delta_root:dependencies+=['--delta-root',args.delta_root]
    if args.twohop_data:dependencies+=['--twohop-data',args.twohop_data]
    if args.initial:dependencies+=['--initial',args.initial]
    for seed in args.seeds:
        for method in args.methods:
            out=root/method/f'seed_{seed}'
            command=['train','--config',args.config,'--diagnostics',args.diagnostics,'--monitor',args.monitor,
                     '--calibration',args.calibration,'--target-train',args.target_train,'--eval-data',args.test,
                     '--seed',seed,'--method',method,'--output',out,*dependencies]
            if method=='reprsi':command+=['--rounds',args.rounds]
            else:command+=['--budget-from',root/'reprsi'/f'seed_{seed}'/'summary.json']
            if args.preparation_costs:command+=['--preparation-costs',*args.preparation_costs]
            if args.resume and (out/'progress.json').exists():command+=['--resume']
            invoke(command)
            summary=json.loads((out/'summary.json').read_text())
            if method!='reprsi' and summary['stop_reason']!='budget':raise RuntimeError('Baseline did not reach its declared compute budget')
            conditions=[('test32',args.test,32,False),('greedy',args.test,1,True)]
            if args.hard_test:conditions.append(('hard128',args.hard_test,128,False))
            for name,data,samples,greedy in conditions:
                destination=out/(name+'.jsonl')
                command=['evaluate','--config',args.config,'--data',data,'--checkpoint',out/summary.get('evaluation_checkpoint','student.pt'),
                         '--seed',seed,'--method',method,'--samples',samples,'--output',destination,*dependencies]
                if greedy:command+=['--greedy']
                if args.resume and destination.with_suffix('.state').exists():command+=['--resume']
                invoke(command)
    for name in ['test32','greedy']+(['hard128'] if args.hard_test else []):
        for method in args.methods:
            command=['aggregate','--inputs',*[root/method/f'seed_{s}'/(name+'.summary.json') for s in args.seeds],
                     '--output',root/'reports'/(method+'_'+name+'.json')]
            if 'direct' in args.methods and method!='direct':command+=['--baseline',*[root/'direct'/f'seed_{s}'/(name+'.summary.json') for s in args.seeds]]
            invoke(command)

    if 'target' in args.methods:
        invoke(['efficiency','--runs',*[root/m/f'seed_{s}' for m in args.methods for s in args.seeds],
                '--reference-method','target','--output',root/'reports/efficiency.json'])


if __name__=='__main__':main()
