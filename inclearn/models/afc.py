#
#
#
#
#   Gradcam and avg ssim values
#
#
#
#




import collections
import copy
import logging
import math

import numpy as np
import torch
from torch.nn import functional as F
import torch.nn as nn

from inclearn.lib import data, factory, losses, network, utils
from inclearn.lib.data import samplers
from inclearn.models.icarl import ICarl
import os
import cv2

logger = logging.getLogger(__name__)

class AFC(ICarl):

    def __init__(self, args):

        #Additional instance variables for mini batch implementation of cka
        #self._mini_batch_counter =  0
        
        self._cka_loss_list = []
        self._cka_iters = args["cka_iters"]
        self._alpha, self._beta, self._gamma = args["alpha"], args["beta"], args["gamma"]

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

        self._pod_flat_config = args.get("pod_flat", {})
        self._pod_spatial_config = args.get("pod_spatial", {})

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
            elif self._finetuning_config["sampling"] == "oversampling":
                _, loader = self.inc_dataset.get_custom_loader(
                    list(range(self._n_classes - self._task_size, self._n_classes)),
                    memory=self.get_memory(),
                    mode="train",
                    sampler=samplers.MemoryOverSampler
                )

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
        #self._update_importance(train_loader)


    def _update_importance(self, train_loader):
        if len(self._multiple_devices) > 1:
            logger.info("Duplicating model on {} gpus.".format(len(self._multiple_devices)))
            training_network = nn.DataParallel(self._network, self._multiple_devices)
        else:
            training_network = self._network

        training_network.convnet.reset_importance()
        training_network.convnet.start_cal_importance()
        for i, input_dict in enumerate(train_loader):
            inputs, targets = input_dict["inputs"], input_dict["targets"]
            memory_flags = input_dict["memory_flags"]

            inputs, targets = inputs.to(self._device), targets.to(self._device)
            outputs = training_network(inputs)

            logits = outputs["logits"]
            if self._post_processing_type is None:
                scaled_logits = self._network.post_process(logits)
            else:
                scaled_logits = logits * self._post_processing_type
            if self._nca_config:
                nca_config = copy.deepcopy(self._nca_config)
                if self._network.post_processor:
                    nca_config["scale"] = self._network.post_processor.factor

                loss = losses.nca(
                    logits,
                    targets,
                    memory_flags=memory_flags,
                    **nca_config
                )

            elif self._softmax_ce:
                # Classification loss is cosine + learned factor + softmax:
                loss = F.cross_entropy(scaled_logits, targets)


            loss.backward()

        training_network.convnet.stop_cal_importance()
        training_network.convnet.normalize_importance()

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
            if self._task >= 1:
                print(f"making old_model_copy")
                self._old_model_copy = copy.deepcopy(self._old_model).to(self._device)
            self._old_model = self._network.copy().eval().to(self._device)
            self._network.on_task_end()

            self._network.set_gradcam_hook()
            self._old_model.set_gradcam_hook()
        else:
            super()._after_task(inc_dataset)

    def _eval_task(self, test_loader): #modified eval_task for avg ssim and gradcam viz
        
        if self._evaluation_type in ("icarl", "nme"):
            return super()._eval_task(test_loader)

        elif self._evaluation_type in ("softmax", "cnn"):
            ''' 
            if self._args["gradcam_viz"]:
                max_class = sum(self.inc_dataset.increments[:self._task+1])
                avg_ssim_per_class = torch.zeros((max_class, self._network.convnet.out_dim), device=self._device).detach()
                print(f"Performing Gradcam also on classes 0 - {max_class}")
                self._network.convnet.activate_gradcam_hooks()
                gradcam_output_folder = self.make_gradcam_folder(self._args, max_class) + str(self._task) + "/"            
            '''
            ypred = []
            ytrue = []
            sample_num = 0
            for batch_num, input_dict in enumerate(test_loader):
                ytrue.append(input_dict["targets"].numpy())

                inputs = input_dict["inputs"].to(self._device)
                targets = input_dict["targets"].to(self._device)
                outputs = self._network(inputs)
                logits = outputs["logits"].detach()
                
                '''
                if self._args['gradcam_viz']:
                    print(f"batch - {batch_num}")
                    sample_num = self.gradcam_viz(input_dict, outputs, sample_num, gradcam_output_folder) 
                    
                    if self._task >= 1: 
                        atts = outputs["attention"]
                        old_atts = self._old_model_copy(inputs)["attention"]

                        old_features = old_atts[3]
                        features = atts[3]

                        term1_result = self.Term1_func(old_features, features, keep_channel=True).detach()
                        term2_result = self.Term2_func(old_features, features, keep_channel=True).detach()
                        term3_result = self.Term3_func(old_features, features, keep_channel=True).detach()
                        
                        temp = (term1_result ** self._alpha ) * (term2_result ** self._beta) * (term3_result ** self._gamma)

                        for i in targets.unique():
                            avg_ssim_per_class[i.item()] += temp[targets == i].sum(dim = 0).detach()

                #end of gradcam if
                '''

                preds = F.softmax(logits, dim=-1)
                ypred.append(preds.cpu().numpy())

            ''' 
            #normalize the avg_ssim value

            if self._args["gradcam_viz"]: 
               logging.info(f"Storing Summed Channel-wise SSIM values at {gradcam_output_folder}")
               print(f"Storing Summed Channel-wise SSIM values at {gradcam_output_folder}")
               torch.save(avg_ssim_per_class, gradcam_output_folder + "ssim_till_class" + str(max_class) + ".pt") 
            '''

            ypred = np.concatenate(ypred)
            ytrue = np.concatenate(ytrue)
            self._network.convnet.deactivate_gradcam_hooks()


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
        elif self._groupwise_factors == "ucir":
            params = [
                {
                    "params": self._network.convnet.parameters(),
                    "lr": self._lr
                },
                {
                    "params": self._network.classifier.new_weights,
                    "lr": self._lr
                },
            ]
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
    
    def _hsic_unbiased(self,K, L):
        """Compute the unbiased estimator of HSIC."""
        n = K.size(0)
        K = self._zero_diag(K)
        L = self._zero_diag(L)
    
        term1 = torch.trace(K @ L)
        term2 = (K.sum() * L.sum()) / ((n - 1) * (n - 2))
        term3 = 2 * (K.sum(dim=0 , keepdim = True) @ L.sum(dim=1 , keepdim = True)) / (n - 2)
    
        hsic = (term1 + term2 - term3) / (n * (n - 3))
    
        return hsic
    
    def _cka_loss(self ,X,Y) :
    
        K = X.flatten(1)
        L = Y.flatten(1)
        
        K = K @ K.T
        L = L @ L.T
    
        #k = self._n_batches if (self._cka_iters  > self._n_batches) else self._cka_iters
        
        if (self._cka_iters == 0) or (self._cka_iters  > self._n_batches)  :
            k = self._n_batches
        else :
            k = self._cka_iters
        
        sum_last_k_lists = lambda lst, k: [sum(x) for x in zip(*lst[-k:])]
        
        self._cka_loss_list.append([ self._hsic_unbiased(K,L), self._hsic_unbiased(K,K) , self._hsic_unbiased(L,L) ])


        if len(self._cka_loss_list) == self._n_batches :
          ckaTerms = sum_last_k_lists(self._cka_loss_list,  1) if k == 1 else sum_last_k_lists(self._cka_loss_list,  self._n_batches % k)
          self._cka_loss_list.clear()

        elif  len(self._cka_loss_list) % k == 0 and len(self._cka_loss_list) != 0 :
          ckaTerms = sum_last_k_lists(self._cka_loss_list,k)
          #self._cka_loss_list.clear()

        else :
          return torch.tensor(0.0)
    
        #Calculate square of CKA only if we have all the denominator terms positive else there could be a possibility of complex number  
        if( ckaTerms[1] > 0 and ckaTerms[2] > 0 ) :
          return (ckaTerms[0]**2 /((ckaTerms[1])*(ckaTerms[2]))) 
        else:
          return torch.tensor([0.0])
    
    
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
                old_importance = old_outputs["importance"]

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
        alpha = self._alpha
        beta = self._beta
        gamma = self._gamma
        factor = self._feature_distil["scheduled_factor"] * math.sqrt(
                        self._n_classes / self._task_size
                    ) 
        if self._old_model is not None:
            if self._feature_distil:

                for i in range(3,len(old_atts)):

                    old_features = old_atts[i]
                    features = atts[i]
                    
                    term1_result = self.Term1_func(old_features, features)
                    term2_result = self.Term2_func(old_features, features)
                    term3_result = self.Term3_func(old_features, features)
                    
                    temp = (term1_result ** alpha ) * (term2_result ** beta) * (term3_result ** gamma)
                    ssim_val = ssim_val + temp
                
                #ssim_val = ssim_val / (len(atts))
                loss += factor * (1. - ssim_val) / 2
                self._metrics["ssim_loss"] += factor * (1. - ssim_val.item()) / 2        
        return loss
        
    def Term1_func(self, old_feat, feat, keep_channel=False):

        #shape of old_feat and feat is b x c x h x w
        
        mu1 = old_feat.mean(dim=(2, 3), keepdim=True) #old_feat/feat shape: b x c x 1 x 1
        mu2 = feat.mean(dim=(2, 3), keepdim=True)
        
        luminance_map = (2 * mu1 * mu2 + 1e-05) / (mu1 ** 2 + mu2 ** 2 + 1e-05) 
        
        if keep_channel:
            return luminance_map.squeeze()
        
        luminance_map = luminance_map.mean(dim=1)  # b x 1 x 1

        luminance_term = luminance_map.squeeze() #b

        return luminance_term.mean() 

 

    def Term2_func(self, old_feat, feat, keep_channel=False):
        
        mu1 = old_feat.mean(dim=(2, 3), keepdim=True)
        mu2 = feat.mean(dim=(2, 3), keepdim=True)

        sigma1_sq = ((old_feat - mu1) ** 2).mean(dim=(2, 3), keepdim=True)
        sigma2_sq = ((feat - mu2) ** 2).mean(dim=(2, 3), keepdim=True)
        
        contrast_map = (2 * torch.sqrt(sigma1_sq) * torch.sqrt(sigma2_sq) + 1e-05) / (sigma1_sq + sigma2_sq + 1e-05)

        if keep_channel:
            return contrast_map.squeeze() # bxc
        
        contrast_map = contrast_map.mean(dim=1) 
        contrast_term = contrast_map.squeeze() #b

        return contrast_term.mean()  

        


    def Term3_func(self,old_feat, feat, keep_channel=False):
       
        #shape of old_feat and feat is b x c x h x w
         
        mu1 = old_feat.mean(dim=(2, 3), keepdim=True)   #b x c x 1 x 1
        mu2 = feat.mean(dim=(2, 3), keepdim=True)

        sigma1_sq = ((old_feat - mu1) ** 2).mean(dim=(2, 3), keepdim=True)  #b x c x 1 x 1
        sigma2_sq = ((feat - mu2) ** 2).mean(dim=(2, 3), keepdim=True)

        sigma12 = ((old_feat - mu1) * (feat - mu2)).mean(dim=(2, 3), keepdim=True) #b x c x 1 x 1

        struc_term_per_samp = (sigma12 + 1e-05) / (torch.sqrt(sigma1_sq * sigma2_sq) + 1e-05)

        if keep_channel:
            return struc_term_per_samp.squeeze()

        struc_mean_across_channels = struc_term_per_samp.mean(dim=1)   #b x 1 x 1
        structure_term = struc_mean_across_channels.squeeze()  # b
        return structure_term.mean()  
         
        
    
    def _scale_compen_cka_loss_M(self,old_feat, feat):
        
        X = old_feat.unsqueeze( dim =1)
        Y = feat.unsqueeze( dim =1)
        X = torch.bmm(X.permute(0,2,1), X)   # dim = b x c x c
        Y = torch.bmm(Y.permute(0,2,1), Y)
        
        C_X = X.std(dim = (1,2))
        C_Y = Y.std(dim = (1,2))
        
        c_2 = 1e-5
        
        return ((2 * C_X * C_Y + c_2) / (C_X ** 2 + C_Y**2 + c_2))

        
    def _cka_loss_M(self, num, den_1, den_2):

        return (num /((den_1 ** 0.5)*(den_2 **0.5))) 
        
    
    def _HSIC_M(self, old_feat, feat):
        
        X = old_feat.unsqueeze( dim =1)
        Y = feat.unsqueeze( dim =1)
        X = torch.bmm(X.permute(0,2,1), X)   # dim = b x c x c
        Y = torch.bmm(Y.permute(0,2,1), Y)
        if True:
            X = X.view(old_feat.shape[0], -1)
            Y = Y.view(feat.shape[0], -1)
            
            Z = X * Y
            
            Z = Z.sum(dim = 1)
            
            Z = Z / (old_feat.shape[1] - 1) ** 2
            
            return Z
        
    
    def gradcam_viz(self, inputs, outputs, sample_num, gradcam_output_folder):
        
        print("\nPassing all test data through the model to get gradcam visualizations")
        dataset = self._args["dataset"]
        open_images = ['imagenet', 'imagenet100']

        np_imgs = inputs["np_imgs"]
        targets = inputs["targets"]
        
        activations = outputs["attention"][-1]
        logits = outputs["logits"]
        preds = logits.argmax(dim=1)
        
        for j in range(logits.shape[0]):    # for each sample in batch
            
            logits[j, preds[j]].backward(retain_graph=True)            
            gradients = self._network.convnet.get_gradcam_gradients()[j].unsqueeze(0)
            pooled_gradients = torch.mean(gradients, dim = [0,2,3])
            act = activations[j].unsqueeze(0)
            
            for k in range(act.shape[1]):
                act[:, k, :, :] *= pooled_gradients[k]
            

            #heatmap = torch.mean(act, dim=1).squeeze().detach().cpu() original gradcam
            #heatmap /= torch.max(heatmap) #original gradcam

            heatmap = act.squeeze().detach().cpu()     # channel-wise gradcam, need .cpu() for next line
            heatmap = np.maximum(heatmap, 0)
            max = torch.amax(heatmap, dim=(1, 2)).view(heatmap.shape[0], 1, 1) # channel-wise, gives cx1x1 tensor with max values
            max[max==0] = 1 #to avoid div by zero
            heatmap /= max
            
            heatmap = heatmap.numpy()

            if dataset in open_images:
                img = cv2.imread(np_imgs[j])
            else:
                img = np_imgs[j].numpy()

            a, b = img.shape[1], img.shape[0]
            for channel in range(heatmap.shape[0]): #loop is unavoidable
                channel_heatmap = cv2.resize(heatmap[channel], (a, b))
                channel_heatmap = np.uint8(255 * channel_heatmap)
                channel_heatmap = cv2.applyColorMap(channel_heatmap, cv2.COLORMAP_JET)
                file_name = gradcam_output_folder + str(targets[j].item()) + "/" + "channel_" + str(channel) + "/" 
                file_name = file_name + str(preds[j].item()) + "_" + str(sample_num) + "_channel" + str(channel) + ".jpg" 
                #print(file_name, end="\n\n")
                superimposed_img = channel_heatmap * 0.4 + img
                cv2.imwrite(file_name, superimposed_img)

            sample_num+=1   
        return sample_num


    def make_gradcam_folder(self, args, max_class):
        output_folder_path = "gradcam_visualizations/" + args["label"] + "_" + args["dataset"] + "/"
        if not os.path.exists(output_folder_path):
            if not os.path.exists("gradcam_visualizations"):
                os.mkdir("gradcam_visualizations")
            os.mkdir(output_folder_path)

        os.chdir(output_folder_path)
        os.mkdir(str(self._task))
        os.chdir(str(self._task))
        
        for i in range(max_class):
            os.mkdir(str(i))
            for j in range(self._network.convnet.out_dim):
                os.mkdir(str(i) + "/channel_" + str(j))
        os.chdir("../../../")
        return output_folder_path




class BoundClipper:

    def __init__(self, lower_bound, upper_bound):
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound

    def __call__(self, module):
        if hasattr(module, "mtl_weight"):
            module.mtl_weight.data.clamp_(min=self.lower_bound, max=self.upper_bound)
        if hasattr(module, "mtl_bias"):
            module.mtl_bias.data.clamp_(min=self.lower_bound, max=self.upper_bound)

