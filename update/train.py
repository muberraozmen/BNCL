from loader import *
from model import *
from loss import *
from runner import *
from utils import *

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-data_dir", type=str, default="C:/Users/jcotn/OneDrive/Desktop/XMTC/resources/stackexchange_philosophy/first5000")
    parser.add_argument("-results_dir", type=str, default="C:/Users/jcotn/PycharmProjects/BNCL/update/outputs/stackex/k_clusters")
    parser.add_argument("-supervision", type=int, default=1)
    parser.add_argument("-annotation_ratio", type=int, default=10)
    parser.add_argument("-embeddings_file", type=str, default="./update/inputs/glove.6B.100d.txt")
    parser.add_argument("-k_lambdas", type=int, default=10)
    #parser.add_argument("-similarity_file", type=str, default="./update/inputs/reuters/similarity.pt")
    parser.add_argument("-hierarchy_file", type=str)
    parser.add_argument("-percentile_neg", type=float, default=0.1)
    parser.add_argument("-percentile_pos", type=float, default=0.9)
    parser.add_argument("-batch_size", type=int, default=128)
    parser.add_argument("-num_layers", type=int, default=2)
    parser.add_argument("-num_epochs", type=int, default=30)
    parser.add_argument("-seed", type=int, default=0)
    parser.add_argument("-device")
    args = parser.parse_args()

    set_seed(seed=args.seed)

    if args.device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device

    results_dir = args.results_dir + '/supervision level ' + str(args.supervision) + '/seed ' + str(args.seed) + time.strftime(' on %m.%d.%Y at %H.%M.%S/')
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)
    logger = get_logger(results_dir=results_dir)
    logger.info("Configuration:")
    logger.info(args)

    data = TransformedData(**vars(args))
    if args.k_lambdas != 0:
        clusters, means, counts, k_range = cluster(data.lambdas, args.k_lambdas)
        for i in range(len(clusters)):
            data.lambdas[i] = means[int(clusters[i])]
    train_loader = DataLoader(data, batch_size=args.batch_size, shuffle=False, drop_last=False)
    if data.annotated_idx is not None:
        annotated_data = data.load_annotated()
        annotated_loader = DataLoader(annotated_data, batch_size=args.batch_size, shuffle=False, drop_last=False)
    else:
        annotated_loader = None
    test_entailments, test_contradictions, test_targets = data.load_test()

    base_predictions = 1 * (test_entailments >= test_contradictions)
    base_metrics = evaluation(targets=test_targets.numpy(), predictions=base_predictions.detach().cpu().numpy())
    logger.info("---- Baseline Performance ----")
    for i, (key, value) in enumerate(base_metrics.items()):
        logger.info('{:} = {:}'.format(key, np.round_(value, decimals=4)))

    model = UpdateModel(num_labels=data.num_labels, adj=data.adj,  num_layers=args.num_layers)
    loss = CollectiveLoss(kappa=data.kappa, lambdas=data.lambdas)
    runner = ModelRunner(model, loss, device=device)

    losses = []
    metrics_best = None
    results = {}
    for epoch in range(args.num_epochs):
        logger.info("---- Epoch {:} -----".format(epoch))
        epoch_loss = runner.train_epoch(train_loader=train_loader, annotated_loader=annotated_loader)
        logger.info("- average training loss = {:.0f}".format(epoch_loss))
        losses.append(epoch_loss)
        test_predictions = runner.test(test_entailments, test_contradictions)
        metrics = evaluation(targets=test_targets.numpy(), predictions=test_predictions.detach().cpu().numpy())

        logger.info("- test performance:")
        for i, (key, value) in enumerate(metrics.items()):
            logger.info('{:} = {:}'.format(key, np.round_(value, decimals=4)))

        if metrics_best is None:
            metrics_best = metrics
            for i, (key, value) in enumerate(metrics.items()):
                torch.save({'targets': test_targets.detach().cpu().numpy(),
                            'predictions': test_predictions.detach().cpu().numpy()},
                           results_dir + key + '_predictions.pt')
                runner.save_model(results_dir + key + '_best.model')
                metrics_best[key] = value
        else:
            for i, (key, value) in enumerate(metrics.items()):
                if metrics[key] >= metrics_best[key]:
                    torch.save({'targets': test_targets.detach().cpu().numpy(),
                                'predictions': test_predictions.detach().cpu().numpy()},
                               results_dir + key + '_predictions.pt')
                    runner.save_model(results_dir + key + '_best.model')
                    metrics_best[key] = value

    logger.info("---- Overall Performance -----")
    for i, (key, value) in enumerate(metrics_best.items()):
        logger.info('- {:} = {:}'.format(key, np.round_(value, decimals=4)))

    if args.k_lambdas != 0:
        f = open(args.results_dir + "/lambda_cluster_results", "a")
        f.write("supervision: "+str(args.supervision)+" k_lambda: "+str(args.k_lambdas))
        f.close()
        for i, (key, value) in enumerate(metrics_best.items()):
            # logger.info('- {:} = {:}'.format(key, np.round_(value, decimals=4)))
            f = open(args.results_dir+"/lambda_cluster_results", "a")
            f.write(' - {:} = {:}'.format(key, np.round_(value, decimals=4)))
            f.close()
        f = open(args.results_dir+"/lambda_cluster_results", "a")
        f.write("\n")
        f.close()

