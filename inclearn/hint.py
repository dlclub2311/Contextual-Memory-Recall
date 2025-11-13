import torch
import torch.nn as nn
import torch.nn.functional as F
import random
import copy
import collections
import os

from inclearn.lib import utils

def dist_between_ftmaps(current_ftmap, avg_ftmap):
    dist = torch.sqrt(((current_ftmap - avg_ftmap)**2).sum(dim = (-1, -2)))
    return torch.sort(dist, descending=True)

def replace_ft_maps(task_id, run_id, current_ftmaps, avg_ftmaps, target, num_channels, p, mode=None):

    if mode["option"]=="random":
        selected_indices = random.sample(range(num_channels), int(p*num_channels))
    elif mode["option"]=="gradcam":
        imp_load = "./gradcam_feat_imp_" + mode["model"] + "_" + mode["label"] + "_run" + str(run_id) + ".pt"
        importances = torch.load(imp_load)
        feature_importances_gt = importances[task_id][target]
        _, top_features_indices = torch.sort(feature_importances_gt, descending=True)
        selected_indices = top_features_indices[:int(p*num_channels)]
    if len(selected_indices) != 0:
        current_ftmaps[selected_indices] = avg_ftmaps[selected_indices]

    return current_ftmaps, selected_indices

def replace_model_weights(task_num, increments, model, task_model, selected_indices, gt_index):
    weights_task = task_model._network.classifier._weights[task_num]
    for i in gt_index:
        i = i - sum(increments[:task_num])
        model._network.classifier._weights[task_num][i, selected_indices] = weights_task[i, selected_indices]

def classifier_broken(model, feature_map):
    features = model._network.convnet.end_features(feature_map)
    features = torch.transpose(features, 0, 1)
    classifier_outputs = model._network.classifier(features)
    return classifier_outputs["logits"]

def get_output_size(model, dataset):
    dataset = dataset.lower()
    print(f"dataset {dataset}")
    if dataset == "cifar100":
        Input = torch.randn((3, 32, 32), device=model._device)
    elif dataset in ["imagenet", "imagenet1000", "imagenet100"]:
        Input = torch.randn((3, 224, 224), device=model._device)

    output = model._network.convnet(Input.unsqueeze(0))["attention"][-1]
    return output.shape

def compute_avg_ftmaps(class_indexes, model, inc_dataset, compute_avg=None, agg=None):
    for class_index in class_indexes:

            class_embeddings = []
            _, loader = inc_dataset.get_custom_loader([class_index])

            for input_dict in loader:
                inputs, targets = input_dict["inputs"].to(model._device), input_dict["targets"].to(model._device)

                outputs = model._network(inputs)
                batch_features = outputs["attention"][-1]

                if compute_avg=="selective":
                    # using only the correct preds to compute average feature map
                    _, preds = (F.softmax(outputs["logits"], dim=1)).max(dim=1)
                    index = (preds == targets)
                    class_embeddings.append(batch_features[index])
                elif compute_avg=="all":
                    class_embeddings.append(batch_features)
            class_embeddings = torch.cat(class_embeddings, dim=0)
            if agg=="mean":
                class_features = torch.mean(class_embeddings, dim=0)
            if agg=="sum":
                class_features = torch.sum(class_embeddings, dim=0)
            model.avg_ft_maps[class_index] = class_features

def compute_gradcam_feat_imp(model, test_loader, device, task_id, feat_imp):
    model._network.convnet.activate_gradcam_hooks()
    for i, input_dict in enumerate(test_loader):
        print(f"\r batch - {i}")
        inputs, targets = input_dict["inputs"].to(device), input_dict["targets"].to(device)
        outputs = model._network(inputs)
        logits = outputs["logits"]

        for j in range(logits.shape[0]):

            logits[j, targets[j]].backward(retain_graph=True)
            gradients = model._network.convnet.get_gradcam_gradients()[j].unsqueeze(0).detach()
            pooled_gradients = torch.mean(gradients, dim = [0,2,3]).detach()
            feat_imp[task_id][targets[j].item()] += pooled_gradients.cpu().detach()
    model._network.convnet.deactivate_gradcam_hooks()
    return feat_imp

def hint_metric(model, run_id, old_class_indices, inc_dataset, task_id, p, mode, logger=None):   
    logger.info(f"replace_prob {p} replace_mode {mode}")
    topk=mode["topk"] 
    models_all_task = []
    for i in range(task_id):
        load_path = "all_tasks_saved/" + mode["label"] + "/" + mode["dataset"] + "/" + "net_" + str(run_id) + "_task_" + str(i) + ".pth"
        task_model = copy.deepcopy(model)
        task_model._network.load_state_dict(torch.load(load_path), strict=False)
        models_all_task.append(task_model)

    num_channels = model._network.convnet.out_dim

    avg_diff_prob = 0.
    per_class_difference = collections.defaultdict(list)
    
    max_old_class = sum(inc_dataset.increments[:task_id])
    task_num=0
    for class_idx in old_class_indices:
        if class_idx==sum(inc_dataset.increments[:task_num+1]):
            task_num+=1

        diff_prob = 0.
        data, test_loader_class_idx = inc_dataset.get_custom_loader([class_idx], mode="test", data_source="test")
        Original_CC,  Original_MC = 0, 0
        Revised_CC, Revised_MC = 0, 0
        logits_replaced_gt = []
        logits_gt = []
        revised_predictions = []
        original_predictions = []

        Targets = []

        for input_dict in test_loader_class_idx:
            
            inputs, targets = input_dict["inputs"].to(model._device), input_dict["targets"].to(model._device)

            outputs = model._network(inputs)
            Targets.append(targets)

            # original probability values
            logit_ground_truth = F.softmax(outputs["logits"], dim=1)[torch.arange(inputs.size(0)), targets]
            logits_gt.append(logit_ground_truth)
            features = outputs["attention"][-1]
            
            _, original_pred = torch.topk(F.softmax(outputs["logits"], dim=1), k=topk, dim=1)
            original_predictions.append(original_pred)

            for i in range(inputs.shape[0]):
                replaced_feature_map, selected_indices = replace_ft_maps(task_id, run_id, features[i], model.avg_ft_maps[targets[i]], targets[i], num_channels, p, mode)
                if len(selected_indices) != 0:
                    replace_model_weights(task_num, inc_dataset.increments, model, models_all_task[task_num], selected_indices, [targets[i]])
                replaced_classifier_outputs = F.softmax(classifier_broken(model, replaced_feature_map), dim=1)
                logits_replaced_gt.append(replaced_classifier_outputs[:, targets[i]])
                _, revised_pred = torch.topk(replaced_classifier_outputs, k=topk, dim=1)
                revised_predictions.append(revised_pred)
        revised_predictions = torch.cat(revised_predictions)
        original_predictions = torch.cat(original_predictions)
        Targets = torch.cat(Targets)

        Revised_CC += (torch.logical_and(original_predictions != Targets.unsqueeze(1), revised_predictions == Targets.unsqueeze(1)).sum()).item()
        Revised_MC += (torch.logical_and(original_predictions == Targets.unsqueeze(1), revised_predictions != Targets.unsqueeze(1)).sum()).item()
        Original_CC += ((original_predictions == Targets.unsqueeze(1)).sum()).item()
        Original_MC += ((original_predictions != Targets.unsqueeze(1)).sum()).item()

        # replaced probability values
        logits_replaced_gt = torch.cat(logits_replaced_gt)
        logits_gt = torch.cat(logits_gt)
        # relative probability
        diff_prob += ((logits_replaced_gt / (logits_gt + 1e-5)).sum()).item() 

        Revised_CC_prop = Revised_CC / (Original_MC + 1e-5)
        Revised_MC_prop = Revised_MC / (Original_CC + 1e-5)
        per_class_difference[class_idx] = (diff_prob)/len(data) * (1 + Revised_CC_prop - Revised_MC_prop)

        logger.info(f"class_idx {class_idx} revised_cc {Revised_CC} revised_mc {Revised_MC} original_cc {Original_CC} original_mc {Original_MC}")

    avg_diff_prob = sum(per_class_difference.values())/len(per_class_difference.values())
    return avg_diff_prob, per_class_difference
