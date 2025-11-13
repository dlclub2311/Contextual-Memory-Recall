import copy
import logging
import math

import numpy as np
import torch
from torch.nn import functional as F
import torch.nn as nn

from inclearn.lib import data, factory, losses, network, utils
from inclearn.models.icarl import ICarl

logger = logging.getLogger(__name__)

class S2IL(ICarl):

    def __init__(self, args):


        self._class_means = []
        
        self._cka_loss_list = []
        self._p, self._q, self._r = args["ssim_p"], args["ssim_q"], args["ssim_r"]

        #--------------------------------------------------------------

        self._disable_progressbar = args.get("no_progressbar", False)

        self._device = args["device"][0]
        self._multiple_devices = args["device"]

        # Optimization:
        self._batch_size = args["batch_size"]
        self._opt_name = args["optimizer"]
        self._lr = args["lr"]
        self._weight_decay = args["weight_decay"]
        self._n_epochs = args["epochs"]
        self._scheduling = args["scheduling"]
        self._lr_decay = args["lr_decay"]

        # Rehearsal Learning:
        self._memory_size = args["memory_size"]
        self._fixed_memory = args["fixed_memory"]
        self._herding_selection = args.get("herding_selection", {"type": "icarl"})
        self._n_classes = 0
        self._last_results = None
        self._validation_percent = args.get("validation")

        self._feature_distil = args.get("feature_distil", {})

        self._nca_config = args.get("nca", {})
        self._softmax_ce = args.get("softmax_ce", False)

        self._perceptual_features = args.get("perceptual_features")
        self._perceptual_style = args.get("perceptual_style")

        self._groupwise_factors = args.get("groupwise_factors", {})
        self._groupwise_factors_bis = args.get("groupwise_factors_bis", {})

        self._class_weights_config = args.get("class_weights_config", {})

        self._evaluation_type = args.get("eval_type", "icarl")
        self._evaluation_config = args.get("evaluation_config", {})

        self._eval_every_x_epochs = args.get("eval_every_x_epochs")
        self._early_stopping = args.get("early_stopping", {})

        self._gradcam_distil = args.get("gradcam_distil", {})

        classifier_kwargs = args.get("classifier_config", {})
        self._network = network.BasicNet(
            args["convnet"],
            convnet_kwargs=args.get("convnet_config", {}),
            classifier_kwargs=classifier_kwargs,
            postprocessor_kwargs=args.get("postprocessor_config", {}),
            device=self._device,
            return_features=True,
            extract_no_act=True,
            classifier_no_act=args.get("classifier_no_act", True),
            attention_hook=True,
            gradcam_hook=bool(self._gradcam_distil)
        )

        self._examplars = {}
        self._means = None

        self._old_model = None

        self._finetuning_config = args.get("finetuning_config")

        self._herding_indexes = []

        self._weight_generation = args.get("weight_generation")

        self._meta_transfer = args.get("meta_transfer", {})
        if self._meta_transfer:
            assert "mtl" in args["convnet"]

        self._post_processing_type = None
        self._data_memory, self._targets_memory = None, None

        self._args = args
        self._args["_logs"] = {}

    @property
    def _memory_per_class(self):
        """Returns the number of examplars per class."""
        if self._fixed_memory:
            return self._memory_size // self._total_n_classes
        return self._memory_size // self._n_classes

    def _train_task(self, train_loader, val_loader):
        if self._meta_transfer:
            logger.info("Setting task meta-transfer")
            self.set_meta_transfer()

        for p in self._network.parameters():
            if p.requires_grad:
                p.register_hook(lambda grad: torch.clamp(grad, -5., 5.))

        logger.debug("nb {}.".format(len(train_loader.dataset)))

        if self._meta_transfer.get("clip"):
            logger.info(f"Clipping MTL weights ({self._meta_transfer.get('clip')}).")
            clipper = BoundClipper(*self._meta_transfer.get("clip"))
        else:
            clipper = None
        self._training_step(
            train_loader, val_loader, 0, self._n_epochs, record_bn=True, clipper=clipper
        )

        self._post_processing_type = None

        if self._finetuning_config and self._task != 0:
            logger.info("Fine-tuning")
            if self._finetuning_config["scaling"]:
                logger.info(
                    "Custom fine-tuning scaling of {}.".format(self._finetuning_config["scaling"])
                )
                self._post_processing_type = self._finetuning_config["scaling"]

            if self._finetuning_config["sampling"] == "undersampling":
                self._data_memory, self._targets_memory, _, _ = self.build_examplars(
                    self.inc_dataset, self._herding_indexes
                )
                loader = self.inc_dataset.get_memory_loader(*self.get_memory())


            if self._finetuning_config["tuning"] == "all":
                parameters = self._network.parameters()
            elif self._finetuning_config["tuning"] == "convnet":
                parameters = self._network.convnet.parameters()
            elif self._finetuning_config["tuning"] == "classifier":
                parameters = self._network.classifier.parameters()
            elif self._finetuning_config["tuning"] == "classifier_scale":
                parameters = [
                    {
                        "params": self._network.classifier.parameters(),
                        "lr": self._finetuning_config["lr"]
                    }, {
                        "params": self._network.post_processor.parameters(),
                        "lr": self._finetuning_config["lr"]
                    }
                ]
            else:
                raise NotImplementedError(
                    "Unknwown finetuning parameters {}.".format(self._finetuning_config["tuning"])
                )

            self._optimizer = factory.get_optimizer(
                parameters, self._opt_name, self._finetuning_config["lr"], self.weight_decay
            )
            self._scheduler = None
            self._training_step(
                loader,
                val_loader,
                self._n_epochs,
                self._n_epochs + self._finetuning_config["epochs"],
                record_bn=False
            )

    @property
    def weight_decay(self):
        if isinstance(self._weight_decay, float):
            return self._weight_decay
        elif isinstance(self._weight_decay, dict):
            start, end = self._weight_decay["start"], self._weight_decay["end"]
            step = (max(start, end) - min(start, end)) / (self._n_tasks - 1)
            factor = -1 if start > end else 1

            return start + factor * self._task * step
        raise TypeError(
            "Invalid type {} for weight decay: {}.".format(
                type(self._weight_decay), self._weight_decay
            )
        )

    def _after_task(self, inc_dataset):
        if self._gradcam_distil:
            self._network.zero_grad()
            self._network.unset_gradcam_hook()
            self._old_model = self._network.copy().eval().to(self._device)
            self._network.on_task_end()

            self._network.set_gradcam_hook()
            self._old_model.set_gradcam_hook()
        else:
            super()._after_task(inc_dataset)

    def _eval_task(self, test_loader):
        if self._evaluation_type in ("icarl", "nme"):
            return super()._eval_task(test_loader)
        elif self._evaluation_type in ("softmax", "cnn"):
            ypred = []
            ytrue = []

            for input_dict in test_loader:
                ytrue.append(input_dict["targets"].numpy())

                inputs = input_dict["inputs"].to(self._device)
                logits = self._network(inputs)["logits"].detach()

                preds = F.softmax(logits, dim=-1)
                ypred.append(preds.cpu().numpy())

            ypred = np.concatenate(ypred)
            ytrue = np.concatenate(ytrue)

            self._last_results = (ypred, ytrue)

            return ypred, ytrue
        else:
            raise ValueError(self._evaluation_type)

    def _gen_weights(self):
        if self._weight_generation:
            utils.add_new_weights(
                self._network, self._weight_generation if self._task != 0 else "basic",
                self._n_classes, self._task_size, self.inc_dataset
            )

    def _before_task(self, train_loader, val_loader):
        self._gen_weights()
        self._n_classes += self._task_size
        logger.info("Now {} examplars per class.".format(self._memory_per_class))

        if self._groupwise_factors and isinstance(self._groupwise_factors, dict):
            if self._groupwise_factors_bis and self._task > 0:
                logger.info("Using second set of groupwise lr.")
                groupwise_factor = self._groupwise_factors_bis
            else:
                groupwise_factor = self._groupwise_factors

            params = []
            for group_name, group_params in self._network.get_group_parameters().items():
                if group_params is None or group_name == "last_block":
                    continue
                factor = groupwise_factor.get(group_name, 1.0)
                if factor == 0.:
                    continue
                params.append({"params": group_params, "lr": self._lr * factor})
                print(f"Group: {group_name}, lr: {self._lr * factor}.")
        
        else:
            params = self._network.parameters()

        self._optimizer = factory.get_optimizer(params, self._opt_name, self._lr, self.weight_decay)

        self._scheduler = factory.get_lr_scheduler(
            self._scheduling,
            self._optimizer,
            nb_epochs=self._n_epochs,
            lr_decay=self._lr_decay,
            task=self._task
        )

        if self._class_weights_config:
            self._class_weights = torch.tensor(
                data.get_class_weights(train_loader.dataset, **self._class_weights_config)
            ).to(self._device)
        else:
            self._class_weights = None
    
    def _zero_diag(self,K):
        """Sets the diagonal elements of a matrix to zero."""
        K = K.clone()
        K.fill_diagonal_(0)
        return K

    def _compute_loss(self, inputs, outputs, targets, onehot_targets, memory_flags):

        #Get features from previously saved model
        features, logits, atts = outputs["raw_features"], outputs["logits"], outputs["attention"]

        if self._post_processing_type is None:
            scaled_logits = self._network.post_process(logits)
        else:
            scaled_logits = logits * self._post_processing_type

        if self._old_model is not None:
            with torch.no_grad():
                old_outputs = self._old_model(inputs)
                old_features = old_outputs["raw_features"]
                old_atts = old_outputs["attention"]

        if self._nca_config:
            nca_config = copy.deepcopy(self._nca_config)
            if self._network.post_processor:
                nca_config["scale"] = self._network.post_processor.factor

            loss = losses.nca(
                logits,
                targets,
                memory_flags=memory_flags,
                class_weights=self._class_weights,
                **nca_config
            )
            
            
            self._metrics["nca"] += loss.item()
        elif self._softmax_ce:
            loss = F.cross_entropy(scaled_logits, targets)
            self._metrics["cce"] += loss.item()
            

        # --------------------
        # Distillation losses:
        # --------------------
        
        ssim_val = torch.tensor(0.,requires_grad = True, device = self._device)
        p = self._p
        q = self._q
        r = self._r
        factor = self._feature_distil["scheduled_factor"] * math.sqrt(
                        self._n_classes / self._task_size
                    ) 
        if self._old_model is not None:
            if self._feature_distil:
                for i in range(0,len(old_atts)): # changed from 3 to 0

                    old_features = old_atts[i]
                    features = atts[i]                    
                    term1_result = self.Term1_func(old_features, features)
                    term2_result = self.Term2_func(old_features, features)
                    term3_result = self.Term3_func(old_features, features)
                    
                    temp = (term1_result ** p) * (term2_result ** q) * (term3_result ** r)
                    ssim_val = ssim_val + temp
                ssim_val = ssim_val/len(old_atts)
                loss += factor * (1. - ssim_val) / 2
                self._metrics["ssim_loss"] += factor * (1. - ssim_val.item()) / 2        
        return loss
        
    def Term1_func(self, old_feat, feat):

        #shape of old_feat and feat is b x c x h x w
        
        mu1 = old_feat.mean(dim=(2, 3), keepdim=True) #old_feat/feat shape: b x c x 1 x 1
        mu2 = feat.mean(dim=(2, 3), keepdim=True)
        
        luminance_map = (2 * mu1 * mu2 + 1e-05) / (mu1 ** 2 + mu2 ** 2 + 1e-05) 
        
        luminance_map = luminance_map.mean(dim=1)  # b x 1 x 1

        luminance_term = luminance_map.squeeze() #b

        return luminance_term.mean() 

 

    def Term2_func(self, old_feat, feat):
        
        mu1 = old_feat.mean(dim=(2, 3), keepdim=True)
        mu2 = feat.mean(dim=(2, 3), keepdim=True)

        sigma1_sq = ((old_feat - mu1) ** 2).mean(dim=(2, 3), keepdim=True)
        sigma2_sq = ((feat - mu2) ** 2).mean(dim=(2, 3), keepdim=True)
        
        contrast_map = (2 * torch.sqrt(sigma1_sq) * torch.sqrt(sigma2_sq) + 1e-05) / (sigma1_sq + sigma2_sq + 1e-05)

        contrast_map = contrast_map.mean(dim=1) 

        contrast_term = contrast_map.squeeze() #b

        return contrast_term.mean()  

    def Term3_func(self,old_feat, feat):
        #shape of old_feat and feat is b x c x h x w
        mu1 = old_feat.mean(dim=(2, 3), keepdim=True)   #b x c x 1 x 1
        mu2 = feat.mean(dim=(2, 3), keepdim=True)

        sigma1_sq = ((old_feat - mu1) ** 2).mean(dim=(2, 3), keepdim=True)  #b x c x 1 x 1
        sigma2_sq = ((feat - mu2) ** 2).mean(dim=(2, 3), keepdim=True)

        sigma12 = ((old_feat - mu1) * (feat - mu2)).mean(dim=(2, 3), keepdim=True) #b x c x 1 x 1

        struc_term_per_samp = (sigma12 + 1e-05) / (torch.sqrt(sigma1_sq * sigma2_sq) + 1e-05)

        struc_mean_across_channels = struc_term_per_samp.mean(dim=1)   #b x 1 x 1
        structure_term = struc_mean_across_channels.squeeze()  # b

        return structure_term.mean()  

class BoundClipper:

    def __init__(self, lower_bound, upper_bound):
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound

    def __call__(self, module):
        if hasattr(module, "mtl_weight"):
            module.mtl_weight.data.clamp_(min=self.lower_bound, max=self.upper_bound)
        if hasattr(module, "mtl_bias"):
            module.mtl_bias.data.clamp_(min=self.lower_bound, max=self.upper_bound)

