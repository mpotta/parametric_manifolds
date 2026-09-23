#!/usr/bin/env python3
"""
Standalone, CLI-driven version of the training logic in train_sample_rnns.ipynb.
Meant to run as a batch job (e.g. via SLURM) rather than interactively.

Examples
--------
python train_rnn.py --net-type direct    --hidden-size 1500 --out-dir results/direct_1500
python train_rnn.py --net-type gener     --hidden-size 1500 --out-dir results/gener_1500
python train_rnn.py --net-type curric    --hidden-size 1500 --out-dir results/curric_1500
python train_rnn.py --net-type full-rank --hidden-size 1500 --out-dir results/fullrank_1500
"""
import argparse
import json
import os
import time

import numpy as np
import torch

import modules4 as md
import funcs_Sphere as fs


def give_vectors(sigma1, sigma2, s_m1, s_m2, s=1, hidden_units=1500, max_iter=100, bn2=0.5):
    # Structured initial connectivity, copied from Fig1_CSG.py: draws (m1, n1, m2, n2, input)
    # from a correlated Gaussian so that m/n (and the cue input) overlap from the start,
    # instead of being independent random vectors.
    bigSigma = np.zeros((5, 5))  # 2*rank + input
    bigSigma[0, 0] = s_m1
    bigSigma[1, 1] = s_m2
    bigSigma[2, 2] = 1.
    bigSigma[3, 3] = 1.
    bigSigma[4, 4] = s ** 2
    bigSigma[0, 2] = bigSigma[2, 0] = sigma1
    bigSigma[1, 3] = bigSigma[3, 1] = sigma2
    bigSigma[3, 4] = bigSigma[4, 3] = s * bn2

    stop = False
    ite = 0
    while not stop and ite < max_iter:
        if np.min(np.linalg.eigvals(bigSigma)) < 0:
            bigSigma[2, 2] *= 1.1
            bigSigma[3, 3] *= 1.1
            ite += 1
        else:
            bigSigma[2, 2] *= 1.1
            bigSigma[3, 3] *= 1.1
            stop = True

    mean = np.zeros(5)
    error0 = 10.
    X_save = None
    for _ in range(100):
        X = np.random.multivariate_normal(mean, bigSigma, hidden_units)
        empSig = np.dot(X.T, X) / hidden_units
        error = np.std(empSig - bigSigma)
        if error < error0:
            error0 = error
            X_save = X
    X = X_save
    return X[:, 0:2], X[:, 2:4], X[:, 4]


def structured_low_rank_init(hidden_size):
    Mnaive, Nnaive, Inaive = give_vectors(sigma1=0.8, sigma2=0.8, s_m1=1., s_m2=1., hidden_units=hidden_size)
    Is_naive = np.random.randn() * Mnaive[:, 0] + np.random.randn() * Mnaive[:, 1]
    I_naive = np.vstack((Inaive, Is_naive))
    O_naive = np.random.randn() * Mnaive[:, 0] + np.random.randn() * Mnaive[:, 1]

    dtype = torch.FloatTensor
    m_init = torch.from_numpy(Mnaive / np.sqrt(hidden_size)).type(dtype)
    n_init = torch.from_numpy(Nnaive / np.sqrt(hidden_size)).type(dtype)
    wi_init = torch.from_numpy(I_naive).type(dtype)
    wo_init = torch.from_numpy(O_naive[:, np.newaxis] / hidden_size).type(dtype)
    return wi_init, wo_init, m_init, n_init


def build_task():
    dt = 10
    tau = 100
    alpha = dt / tau
    std_noise_rec = np.sqrt(2 * alpha) * 0.1
    input_size = 2
    output_size = 1
    Nt = 350
    R_on = 1000 // dt
    SR_on = 100 // dt
    tss_ms = np.array([800, 1050, 1300, 1550])
    tss = tss_ms // dt
    amps = np.linspace(0, 0.25, len(tss_ms))
    return dict(dt=dt, tau=tau, alpha=alpha, std_noise_rec=std_noise_rec, input_size=input_size,
                output_size=output_size, Nt=Nt, R_on=R_on, SR_on=SR_on, tss_ms=tss_ms, tss=tss, amps=amps)


def train_direct(task, hidden_size, rank, n_epochs, batch_size, trials_train):
    wi_init, wo_init, m_init, n_init = structured_low_rank_init(hidden_size)
    net = md.OptimizedLowRankRNN(
        task['input_size'], hidden_size, task['output_size'], task['std_noise_rec'], task['alpha'],
        rank=rank, train_wi=True, train_wo=True, train_h0=True,
        wi_init=wi_init, wo_init=wo_init, m_init=m_init, n_init=n_init)

    input_train, output_train, mask_train, _, _ = fs.create_inp_out2(
        trials_train, task['Nt'], task['tss'], task['amps'], task['R_on'], task['SR_on'], perc=0.1)

    loss = md.train(
        net, input_train, output_train, mask_train,
        n_epochs=n_epochs, lr=1e-3, batch_size=batch_size, clip_gradient=1.0,
        plot_learning_curve=False, plot_gradient=False, save_loss=True)
    return net, np.asarray(loss)


def train_gener(task, hidden_size, rank, n_epochs, batch_size, trials_train):
    tss_2int = task['tss'][[0, -1]]
    amps_2int = task['amps'][[0, -1]]

    wi_init, wo_init, m_init, n_init = structured_low_rank_init(hidden_size)
    net = md.OptimizedLowRankRNN(
        task['input_size'], hidden_size, task['output_size'], task['std_noise_rec'], task['alpha'],
        rank=rank, train_wi=True, train_wo=True, train_h0=True,
        wi_init=wi_init, wo_init=wo_init, m_init=m_init, n_init=n_init)

    input_train, output_train, mask_train, _, _ = fs.create_inp_out2(
        trials_train, task['Nt'], tss_2int, amps_2int, task['R_on'], task['SR_on'], perc=0.1)

    loss = md.train(
        net, input_train, output_train, mask_train,
        n_epochs=n_epochs, lr=1e-3, batch_size=batch_size, clip_gradient=1.0,
        plot_learning_curve=False, plot_gradient=False, save_loss=True)
    return net, np.asarray(loss)


def train_curric(task, hidden_size, rank, n_epochs_total, n_steps, ts_min, batch_size, trials_train,
                  final_stage_boost=3):
    # Duration-progressive curriculum: all 4 conditions throughout, durations ramped
    # from ts_min fraction up to 100% of the final target intervals over n_steps stages.
    # The final (hardest, full-duration) stage gets `final_stage_boost` times as many
    # epochs as the earlier stages, since it otherwise tends to destabilize (see notebook).
    Tss_curric = fs.gen_intervals(task['tss_ms'], n_steps, ts_min=ts_min)

    wi_init, wo_init, m_init, n_init = structured_low_rank_init(hidden_size)
    net = md.OptimizedLowRankRNN(
        task['input_size'], hidden_size, task['output_size'], task['std_noise_rec'], task['alpha'],
        rank=rank, train_wi=True, train_wo=True, train_h0=True,
        wi_init=wi_init, wo_init=wo_init, m_init=m_init, n_init=n_init)

    weights = [1] * (n_steps - 1) + [final_stage_boost]
    epochs_per_unit = n_epochs_total / sum(weights)

    all_loss = []
    for stage in range(n_steps):
        tss_stage = (Tss_curric[:, stage] // task['dt']).astype(int)
        input_train, output_train, mask_train, _, _ = fs.create_inp_out2(
            trials_train, task['Nt'], tss_stage, task['amps'], task['R_on'], task['SR_on'], perc=0.1)
        n_epochs_stage = max(1, round(epochs_per_unit * weights[stage]))
        loss_stage = md.train(
            net, input_train, output_train, mask_train,
            n_epochs=n_epochs_stage, lr=1e-3, batch_size=batch_size, clip_gradient=1.0,
            plot_learning_curve=False, plot_gradient=False, save_loss=True)
        all_loss.extend(loss_stage)
        print(f"stage {stage} ({Tss_curric[:, stage].astype(int)} ms, {n_epochs_stage} epochs): "
              f"final loss={loss_stage[-1]:.4f}", flush=True)
    return net, np.asarray(all_loss)


def train_full_rank(task, hidden_size, n_epochs, batch_size, trials_train):
    net = md.FullRankRNN(
        task['input_size'], hidden_size, task['output_size'], 0.1 * task['std_noise_rec'], task['alpha'],
        train_wi=True, train_wo=True, train_h0=True)

    input_train, output_train, mask_train, _, _ = fs.create_inp_out2(
        trials_train, task['Nt'], task['tss'], task['amps'], task['R_on'], task['SR_on'], perc=0.1)

    loss = md.train(
        net, input_train, output_train, mask_train,
        n_epochs=n_epochs, lr=1e-4, batch_size=batch_size, clip_gradient=1.0,
        plot_learning_curve=False, plot_gradient=False, save_loss=True)
    return net, np.asarray(loss)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--net-type', required=True, choices=['direct', 'gener', 'curric', 'full-rank'])
    parser.add_argument('--hidden-size', type=int, default=1500)
    parser.add_argument('--rank', type=int, default=2, help='ignored for --net-type full-rank')
    parser.add_argument('--n-epochs', type=int, default=60,
                         help='total epoch budget; for --net-type curric this is split across stages')
    parser.add_argument('--n-steps', type=int, default=5, help='curriculum stages (curric only)')
    parser.add_argument('--ts-min', type=float, default=0.4, help='shortest-stage duration fraction (curric only)')
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--trials-train', type=int, default=200)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--out-dir', required=True)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    os.makedirs(args.out_dir, exist_ok=True)
    task = build_task()

    t0 = time.time()
    if args.net_type == 'direct':
        net, loss = train_direct(task, args.hidden_size, args.rank, args.n_epochs, args.batch_size, args.trials_train)
    elif args.net_type == 'gener':
        net, loss = train_gener(task, args.hidden_size, args.rank, args.n_epochs, args.batch_size, args.trials_train)
    elif args.net_type == 'curric':
        net, loss = train_curric(task, args.hidden_size, args.rank, args.n_epochs, args.n_steps, args.ts_min,
                                  args.batch_size, args.trials_train)
    elif args.net_type == 'full-rank':
        net, loss = train_full_rank(task, args.hidden_size, args.n_epochs, args.batch_size, args.trials_train)
    else:
        raise ValueError(args.net_type)
    elapsed = time.time() - t0

    torch.save(net.state_dict(), os.path.join(args.out_dir, 'net.pt'))
    np.savez(os.path.join(args.out_dir, 'loss.npz'), loss=loss)
    with open(os.path.join(args.out_dir, 'config.json'), 'w') as f:
        json.dump({**vars(args), 'final_loss': float(loss[-1]), 'elapsed_sec': elapsed}, f, indent=2)

    print(f"[{args.net_type}] hidden_size={args.hidden_size} final_loss={loss[-1]:.5f} "
          f"elapsed={elapsed:.1f}s -> {args.out_dir}", flush=True)


if __name__ == '__main__':
    main()
