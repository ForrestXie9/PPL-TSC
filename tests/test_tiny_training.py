import sys
from pathlib import Path
from types import SimpleNamespace
import unittest
import torch
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from agent.tiny_light.tiny_light import TinyLight
from agent.tiny_light.alpha import Alpha


class TinyTrainingTest(unittest.TestCase):
    def test_td_loss_updates_alpha_and_network_uses_fresh_graph(self):
        agent=TinyLight.__new__(TinyLight)
        agent.n_alpha=torch.nn.ModuleList([Alpha(3,{})])
        agent.network_local=torch.nn.Linear(1,1)
        agent.network_target=torch.nn.Linear(1,1)
        agent.network_optim=torch.optim.SGD(agent.network_local.parameters(),lr=.01)
        agent.alpha_optim=torch.optim.SGD(agent.n_alpha.parameters(),lr=.01)
        agent.beta=0.;agent.tau=.1;agent._time_to_learn=lambda:True
        agent.replay_buffer=SimpleNamespace(sample_experience=lambda:(None,)*5)
        seen=[]
        def critic(*args):
            a=agent.n_alpha[0].alpha;seen.append(a.detach().clone())
            return (a*torch.tensor([1.,2.,3.])).sum().square()+agent.network_local.weight.square().sum()
        agent._compute_critic_loss=critic
        agent.learn()
        self.assertEqual(len(seen),2)
        self.assertFalse(torch.equal(seen[0],seen[1]))
        self.assertTrue(torch.isfinite(agent.n_alpha[0].alpha).all())


if __name__=='__main__':unittest.main()
