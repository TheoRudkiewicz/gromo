import argparse

import git
import matplotlib.pyplot as plt
import torch
from tqdm import tqdm

from gromo.graph_network.dag_growing_network import GraphGrowingNetwork


parser = argparse.ArgumentParser()
parser.add_argument("--exp_name", type=str, default="Debug")
parser.add_argument("--job_id", type=str)
parser.add_argument("--node_name", type=str)
parser.add_argument("--iters", type=int, default=5)
parser.add_argument(
    "--parallel_edges", action=argparse.BooleanOptionalAction, default=True
)
parser.add_argument("--neurons", type=int, default=20)
parser.add_argument("--new_opt", action=argparse.BooleanOptionalAction, default=True)
parser.add_argument("--inter_train", action=argparse.BooleanOptionalAction, default=True)
args = parser.parse_args()


def setup_experiment_tags():
    repo = git.Repo(search_parent_directories=True)
    git_commit = repo.head.object.hexsha
    tags = {
        "git.commit": git_commit,
        "slurm.job_id": args.job_id,
        "slurm.node_name": args.node_name,
    }
    return tags


# loss_train, loss_dev, loss_val, loss_test = [], [], [], []
# acc_train, acc_dev, acc_val, acc_test = [], [], [], []
acc_test = []
indices = []


def grow_network(
    net: GraphGrowingNetwork,
    data_rng: torch.Generator,
    steps: int,
    parallel_edges: bool,
    inter_train: bool,
    new_opt: bool,
    verbose: bool = False,
):

    # mlflow.log_param("neurons", neurons)
    # mlflow.log_param("parallel layers", parallel_edges)
    # mlflow.log_param("new optimization", new_opt)
    tags = setup_experiment_tags()
    net.logger.start_run(tags=tags)

    for _ in tqdm(range(steps)):
        print("\nStep", net.global_step + 1)
        net.grow_step(
            generator=data_rng,
            inter_train=inter_train,
            verbose=verbose,
        )
        # Temporary stats
        # loss_train.append(net.growth_loss_train)
        # loss_dev.append(net.loss_dev)
        # loss_val.append(net.loss_val)
        # loss_test.append(net.loss_test)
        # acc_train.append(net.acc_train)
        # acc_dev.append(net.acc_dev)
        # acc_val.append(net.acc_val)
        acc_test.append(net.acc_test)
        indices.append(len(net.hist_loss_dev))

        if verbose:
            print("\n********* NEW GRAPH *********")
            net.dag.draw()

        # try:
        #     graph = DAG_to_pyvis(net.network.G)
        #     pyvis_path = "tmp/graph_.html"
        #     with portalocker.Lock(pyvis_path, timeout=1) as fh:
        #         graph.save_graph(pyvis_path)
        #         net.logger.log_artifact(pyvis_path)
        # except Exception as error:
        #     print(error)
    # print(mlflow.MlflowClient().get_run(run.info.run_id).data)
    net.logger.end_run()
    return net


if __name__ == "__main__":
    net = GraphGrowingNetwork(
        in_features=28 * 28, out_features=10, neurons=args.neurons, exp_name=args.exp_name
    )
    net.load_MNIST_data()
    print(net.dag)
    # net.dag.draw()
    print(net.device)
    print()

    data_rng = torch.Generator()
    grow_network(
        net,
        data_rng=data_rng,
        steps=args.iters,
        inter_train=args.inter_train,
        parallel_edges=args.parallel_edges,
        new_opt=args.new_opt,
        verbose=False,
    )

    fig, ax1 = plt.subplots()
    ax2 = ax1.twinx()
    p1 = ax1.plot(net.hist_loss_dev, label="development loss")
    p2 = ax2.plot(net.hist_acc_dev, label="development accuracy")
    # p3 = ax2.plot(net.hist_acc_val, label="validation accuracy")
    # p4 = ax1.scatter(indices, loss_train, marker='o', label="train loss")
    # p5 = ax1.scatter(indices, loss_test, marker='o', label="test loss")
    # p6 = ax2.scatter(indices, acc_train, marker='^', label="train accuracy")
    p7 = ax2.scatter(indices, acc_test, marker="^", label="test accuracy")
    plots = p1 + p2 + [p7]
    labels = [p.get_label() for p in plots]
    ax1.legend(plots, labels)
    ax1.set_xlabel("epochs of intermediate training")
    ax1.set_ylabel("loss")
    ax2.set_ylabel("accuracy")
    plt.show()
    print()

    # TODO: check correct data partitions
    # TODO: remove print statements
    # TODO: fix documentation
    # TODO: profile memory
    # TODO: compare with simple model
    # TODO: test with cifar