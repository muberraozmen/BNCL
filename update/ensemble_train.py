from loader import *
from model import *
from loss import *
from runner import *
from utils import *

# from utils import ensemble_evaluation

if __name__ == '__main__':
    path = 'C:/Users/jcotn/PycharmProjects/BNCL/'
    parser = argparse.ArgumentParser()
    parser.add_argument("-data_dir", type=str, default=path + "/update/inputs/stackexchange_philosophy")
    parser.add_argument("-results_dir", type=str, default=path + "/update/outputs/reuters/")
    parser.add_argument("-supervision", type=int, default=1)
    parser.add_argument("-annotation_ratio", type=int, default=0)
    parser.add_argument("-embeddings_file", type=str, default=path + "/update/inputs/glove.6B.100d.txt")
    parser.add_argument("-cluster_lambdas", type=int, default=0)
    parser.add_argument("-hierarchy_file", type=str)
    parser.add_argument("-percentile_neg", type=float, default=0.1)
    parser.add_argument("-percentile_pos", type=float, default=0.9)
    parser.add_argument("-batch_size", type=int, default=128)
    parser.add_argument("-num_layers", type=int, default=2)
    parser.add_argument("-num_epochs", type=int, default=30)
    parser.add_argument("-seed", type=int, default=0)
    parser.add_argument("-num_ensembles", type=int, default=20)
    parser.add_argument("-device")
    args = parser.parse_args()

    set_seed(seed=args.seed)
    num_ensembles = args.num_ensembles
    if args.device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device

    results_dir = args.results_dir + '/supervision level ' + str(args.supervision) + '/seed ' + str(
        args.seed) + time.strftime(' on %m.%d.%Y at %H.%M.%S/')
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)
    logger = get_logger(results_dir=results_dir)
    logger.info("Configuration:")
    logger.info(args)

    data = TransformedData(**vars(args))
    if args.cluster_lambdas != 0:
        clusters, means, counts, k_range = cluster(data.lambdas, args.cluster_lambdas)
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

    model = UpdateModel(num_labels=data.num_labels, adj=data.adj, num_layers=args.num_layers)
    # data.lambdas = [torch.rand(135) for i in range(num_ensembles)]
    data.lambdas = torch.load(args.data_dir + '/' + 'lambdas.pt')[-num_ensembles:]
    # data.lambdas = [data.lambdas for i in range(num_ensembles)]
    ensemble_loss = [CollectiveLoss(kappa=data.kappa, lambdas=data.lambdas[i]) for i in range(num_ensembles)]
    # print(data.lambdas[i].dtype)
    runners = [ModelRunner(model, ensemble_loss[i], device=device) for i in range(num_ensembles)]

    losses = [[] for i in range(num_ensembles)]
    metrics_best = [None for i in range(num_ensembles)]
    results = {}
    ensemble_predictions = [[] for i in range(num_ensembles)]
    best_ensemble_predictions = [[] for i in range(num_ensembles)]
    ensemble_metrics_best = None
    for epoch in range(args.num_epochs):
        logger.info("---- Epoch {:} -----".format(epoch))
        for e in range(num_ensembles):

            epoch_loss = runners[e].train_epoch(train_loader=train_loader, annotated_loader=annotated_loader)
            # logger.info("- average training loss = {:.0f}".format(epoch_loss))
            losses[e].append(epoch_loss)
            ensemble_predictions[e] = runners[e].test(test_entailments, test_contradictions).cpu()

            metrics = evaluation(targets=test_targets.numpy(),
                                 predictions=ensemble_predictions[e].detach().cpu().numpy())

            # logger.info("- test performance:")
            # for i, (key, value) in enumerate(metrics.items()):
            #     logger.info('{:} = {:}'.format(key, np.round_(value, decimals=4)))

            if metrics_best[e] is None:
                metrics_best[e] = metrics
                for i, (key, value) in enumerate(metrics.items()):
                    torch.save({'targets': test_targets.detach().cpu().numpy(),
                                'predictions': ensemble_predictions[e].detach().cpu().numpy()},
                               results_dir + key + '_predictions.pt')
                    runners[e].save_model(results_dir + key + '_best.model')
                    metrics_best[e][key] = value
                    best_ensemble_predictions[e] = ensemble_predictions[e]
            else:
                for i, (key, value) in enumerate(metrics.items()):
                    if metrics[key] >= metrics_best[e][key]:
                        torch.save({'targets': test_targets.detach().cpu().numpy(),
                                    'predictions': ensemble_predictions[e].detach().cpu().numpy()},
                                   results_dir + key + '_predictions.pt')
                        runners[e].save_model(results_dir + key + '_best.model')
                        metrics_best[e][key] = value

        ensemble_predictions_run = np.stack(ensemble_predictions, axis=0)
        # print(ensemble_predictions.shape)
        ensemble_predictions_run = np.mean(ensemble_predictions_run, axis=0)
        # print(ensemble_predictions.shape)
        ensemble_predictions_run = np.where(ensemble_predictions_run > 0.5, 1, 0)
        # print(ensemble_predictions.shape)
        ensemble_metrics = ensemble_evaluation(targets=test_targets,
                                               predictions=ensemble_predictions_run)
        average_epoch_loss = 0
        for k in range(num_ensembles):
            average_epoch_loss += losses[k][-1]
        average_epoch_loss /= num_ensembles
        logger.info("- average training loss = {:.0f}".format(average_epoch_loss))
        logger.info("---- Ensemble Performance -----")
        for i, (key, value) in enumerate(ensemble_metrics.items()):
            logger.info('- {:} = {:}'.format(key, np.round_(value, decimals=4)))
        if ensemble_metrics_best is None:
            ensemble_metrics_best = ensemble_metrics
            for i, (key, value) in enumerate(ensemble_metrics.items()):
                torch.save({'ensemble_targets': test_targets.detach().cpu().numpy(),
                            'ensemble_predictions': ensemble_predictions_run},
                           results_dir + key + '_predictions.pt')
                for k in range(num_ensembles):
                    runners[k].save_model(results_dir + key + '_best_ensemble.model' + str(k))
                ensemble_metrics_best[key] = value
                best_ensemble_predictions = ensemble_predictions_run
        else:
            for i, (key, value) in enumerate(ensemble_metrics.items()):
                if ensemble_metrics[key] >= ensemble_metrics_best[key]:
                    torch.save({'ensemble_targets': test_targets.detach().cpu().numpy(),
                                'ensemble_predictions': ensemble_predictions_run},
                               results_dir + key + 'ensemble_predictions.pt')
                    for k in range(num_ensembles):
                        runners[k].save_model(results_dir + key + '_best_ensemble.model' + str(k))
                    ensemble_metrics_best[key] = value
                    best_ensemble_predictions = ensemble_predictions_run

    logger.info("---- Overall Performance -----")
    # ensemble_predictions_run = np.stack(ensemble_predictions, axis=0)
    # # print(ensemble_predictions.shape)
    # ensemble_predictions_run = np.mean(ensemble_predictions_run, axis=0)
    # # print(ensemble_predictions.shape)
    # ensemble_predictions_run = np.where(ensemble_predictions_run > 0.5, 1, 0)
    # # print(ensemble_predictions.shape)
    # overall_metrics = ensemble_evaluation(targets=test_targets,
    #                                        predictions=ensemble_predictions_run)
    for i, (key, value) in enumerate(ensemble_metrics_best.items()):
        logger.info('- {:} = {:}'.format(key, np.round_(value, decimals=4)))

    if args.cluster_lambdas != 0:
        f = open(args.results_dir + "/lambda_cluster_results", "a")
        f.write("supervision: " + str(args.supervision) + " k_lambda: " + str(args.cluster_lambdas))
        f.close()
        for i, (key, value) in enumerate(metrics_best.items()):
            # logger.info('- {:} = {:}'.format(key, np.round_(value, decimals=4)))
            f = open(args.results_dir + "/lambda_cluster_results", "a")
            f.write(' - {:} = {:}'.format(key, np.round_(value, decimals=4)))
            f.close()
        f = open(args.results_dir + "/lambda_cluster_results", "a")
        f.write("\n")
        f.close()

    logger.info("---- Bootstrap Testing-----")
    # logger.info("---- Bootstrap Testing-----")
    # bootstraps = {}
    # for i, (key, value) in enumerate(metrics.items()):
    #     test_results = torch.load(results_dir + key + '_predictions.pt')
    #     y_true = test_results['targets']
    #     y_pred = test_results['predictions']
    #     num_samples = y_true.shape[0]
    #     temp = []
    #     for k in range(1000):
    #         sample_ids = random.choices(range(num_samples), k=100)
    #         metrics = evaluation(targets=test_targets[sample_ids, :],
    #                              predictions=test_predictions[sample_ids, :])
    #         temp.append(metrics[key])
    #     bootstraps[key] = np.asarray(temp)
    # torch.save(bootstraps, results_dir + 'bootstraps.pt')

    bootstraps = {}
    for i, (key, value) in enumerate(metrics.items()):
        test_results = torch.load(results_dir + key + '_predictions.pt')
        y_true = test_results['targets']
        y_pred = test_results['predictions']
        num_samples = y_true.shape[0]
        temp = []
        for k in range(1000):
            sample_ids = random.choices(range(num_samples), k=100)

            ensemble_predictions_run = np.stack(ensemble_predictions, axis=0)
            # print(ensemble_predictions.shape)
            ensemble_predictions_run = np.mean(ensemble_predictions_run, axis=0)
            # print(ensemble_predictions.shape)
            ensemble_predictions_run = np.where(ensemble_predictions_run > 0.5, 1, 0)
            # print(ensemble_predictions.shape)
            metrics = ensemble_evaluation(targets=test_targets[sample_ids, :],
                                          predictions=ensemble_predictions_run[sample_ids, :])
            temp.append(metrics[key])
        bootstraps[key] = np.asarray(temp)
    torch.save(bootstraps, results_dir + 'bootstraps.pt')





