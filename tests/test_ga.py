import copy

import torch

from brain import PopulationBrain
from ga import GATrainer, ga_policy_from_checkpoint
from utils import load_checkpoint


def test_batched_forward_matches_per_car_loop():
    torch.manual_seed(0)
    brain = PopulationBrain(7, (13, 16, 16, 2))
    obs = torch.randn(7, 13)
    out = brain(obs)
    for p in range(7):
        x = obs[p]
        for W, b in brain.layers(brain.genes):
            x = torch.tanh(x @ W[p] + b[p, 0])
        assert torch.allclose(out[p], x, atol=1e-6)
    assert brain.n_genes == 13 * 16 + 16 + 16 * 16 + 16 + 16 * 2 + 2


def small_cfg(cfg, n=20):
    c = copy.deepcopy(cfg)
    c.env.n_cars = n
    c.checkpoint_dir = "checkpoints_test"
    return c


def test_evolve_keeps_elites_and_prefers_fit_parents(cfg):
    tr = GATrainer(small_cfg(cfg), torch.device("cpu"), seed=0)
    genes = tr.brain.genes.clone()
    fitness = torch.arange(20, dtype=torch.float32)           # car 19 is best
    tr.evolve(fitness)
    new = tr.brain.genes
    assert new.shape == genes.shape
    for rank in range(tr.ga.elites):                          # elites copied unchanged, best first
        assert torch.equal(new[rank], genes[19 - rank])
    assert not torch.equal(new[tr.ga.elites:], genes[tr.ga.elites:])

    # Tournament selection: winners skew toward high fitness.
    contenders = torch.randint(0, 20, (10000, 2, tr.ga.tournament_k))
    winners = contenders.gather(2, fitness[contenders].argmax(2, keepdim=True)).float()
    assert winners.mean() > 12


def test_compaction_does_not_change_results(cfg):
    """Dropping crashed cars from the batch must give every car the same result."""
    c = small_cfg(cfg)
    c.ga.max_generation_time = 20.0

    class NoCompact(GATrainer):
        def compact(self):
            pass

    results = []
    for cls in (GATrainer, NoCompact):
        tr = cls(c, torch.device("cpu"), seed=3)
        tr.train_until(max_iters=1)
        results.append((tr.fitness.clone(), tr.env_steps))
    assert torch.allclose(results[0][0], results[1][0], atol=1e-3)
    assert results[0][1] == results[1][1]


def test_generations_run_and_checkpoint_roundtrip(cfg, tmp_path):
    c = small_cfg(cfg)
    c.ga.max_generation_time = 5.0
    tr = GATrainer(c, torch.device("cpu"), seed=0)
    tr.train_until(max_iters=2)
    assert tr.iteration == 2 and len(tr.history) == 2
    assert tr.env_steps > 0 and tr.history[1]["env_steps"] > tr.history[0]["env_steps"]
    path = tr.save(str(tmp_path / "ga.pt"))

    tr2 = GATrainer(c, torch.device("cpu"), seed=1)
    tr2.load(path)
    assert torch.equal(tr2.brain.genes, tr.brain.genes)
    assert tr2.iteration == 2 and tr2.env_steps == tr.env_steps
    policy = ga_policy_from_checkpoint(load_checkpoint(path), "cpu")
    assert policy(torch.zeros(1, 13)).shape == (1, 2)
